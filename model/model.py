from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from model.backbone import TransfuserBackbone
from configs._base import TransfuserConfig
from model.enums import BoundingBox2DIndex, StateSE2Index


class TransfuserModel(nn.Module):
    """Torch module for Transfuser."""

    def __init__(self, config: TransfuserConfig):
        super().__init__()

        # Query 구성: [truck trajectory(1), (선택) trailer trajectory(1), agents(N)]
        # truck/trailer를 별도 query로 분리해 각자 다른 head로 회귀.
        # trailer query는 현재 sample에 ego_trailer 없으면 loss에서 mask=0으로 제외됨.
        # `use_trailer_head=False`이면 trailer slot 자체를 query에서 빼고 head도 빌드 안 함
        # → paper의 strict truck-only baseline용.
        self._query_splits = [1]                     # truck trajectory query
        if config.use_trailer_head:
            self._query_splits.append(1)             # trailer trajectory query
        self._query_splits.append(config.num_bounding_boxes)  # agent detection queries

        self._config = config
        self._backbone = TransfuserBackbone(config)

        # keyval = (lidar BEV tokens after ÷32 stem) + 1 status token.
        # lidar BEV token 수 = lidar_vert_anchors × lidar_horz_anchors. v3·v4의 8×8=64인데
        # range 변경 config (e.g. v6_range_full)는 다른 grid라 동적으로 산출.
        n_keyval = config.lidar_vert_anchors * config.lidar_horz_anchors + 1
        self._keyval_embedding = nn.Embedding(n_keyval, config.tf_d_model)
        self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)

        self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)
        # Status encoder input is concatenated from optional channels:
        #   - ego status (vx, vy, ax, ay) = 4 dims  — toggle: use_ego_status
        #   - driving_command one-hot      = 3 dims  — toggle: use_driving_command
        # VAD §4.2 setup drops ego status to avoid shortcut learning. At least
        # one of the two must be on so the status token has content.
        status_dim = (4 if config.use_ego_status else 0) \
                   + (3 if config.use_driving_command else 0)
        assert status_dim > 0, (
            "Either use_ego_status or use_driving_command must be True; "
            "otherwise the status token has no content."
        )
        self._status_encoding = nn.Linear(status_dim, config.tf_d_model)

        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(
                config.bev_features_channels,
                config.bev_features_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.bev_features_channels,
                config.num_bev_classes,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Upsample(
                size=(
                    config.lidar_resolution_height // 2,
                    config.lidar_resolution_width,
                ),
                mode="bilinear",
                align_corners=False,
            ),
        )

        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )

        self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)
        self._agent_head = AgentHead(
            num_agents=config.num_bounding_boxes,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )

        self._trajectory_head = TrajectoryHead(
            num_poses=config.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )

        # Trailer trajectory head — truck head와 동일 구조, 별도 파라미터.
        # 같은 query feature space를 공유하지만 출력 head를 분리해 truck/trailer를 따로 학습.
        # `use_trailer_head=False`이면 빌드 자체를 skip → 모델 capacity가 truck-only로 줄어듦.
        if config.use_trailer_head:
            self._trailer_trajectory_head = TrajectoryHead(
                num_poses=config.num_poses,
                d_ffn=config.tf_d_ffn,
                d_model=config.tf_d_model,
            )

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        camera_feature: torch.Tensor = features["camera_feature"]
        if self._config.latent:
            lidar_feature = None
        else:
            lidar_feature: torch.Tensor = features["lidar_feature"]
        # Build the status encoder input from optional channels. The status_dropout
        # below only masks the ego-status channels — the driving_command is a
        # navigation signal (not a leakage target) and must always pass through.
        ego_status: torch.Tensor = features["status_feature"]  # (B, 4)
        batch_size = ego_status.shape[0]

        # Status dropout — 학습 시에만, sample-level로 ego status 일부를 0으로 마스킹.
        # 목적: status(vx,vy,ax,ay) → trajectory의 trivial mapping(vx·Δt)에만 의존하지 않고
        #       image/lidar branch가 의미 있게 학습되도록 강제.
        # 마스킹된 0은 학습에 자주 등장하므로 in-distribution이라 inference 영향 없음.
        # use_ego_status=False면 ego status를 아예 사용하지 않으므로 dropout 의미 없음.
        if (
            self._config.use_ego_status
            and self.training
            and self._config.status_dropout_p > 0.0
        ):
            keep_mask = (
                torch.rand(batch_size, 1, device=ego_status.device)
                > self._config.status_dropout_p
            ).to(ego_status.dtype)
            ego_status = ego_status * keep_mask

        parts = []
        if self._config.use_ego_status:
            parts.append(ego_status)
        if self._config.use_driving_command:
            parts.append(features["driving_command"])
        status_feature = torch.cat(parts, dim=-1)

        bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature, lidar_feature)

        bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1)
        bev_feature = bev_feature.permute(0, 2, 1)
        status_encoding = self._status_encoding(status_feature)

        keyval = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1)
        keyval += self._keyval_embedding.weight[None, ...]

        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        query_out = self._tf_decoder(query, keyval)

        # query_splits 순서: [truck, (옵션) trailer, agents]. use_trailer_head 토글에 맞춤.
        splits = query_out.split(self._query_splits, dim=1)
        if self._config.use_trailer_head:
            trajectory_query, trailer_query, agents_query = splits
        else:
            trajectory_query, agents_query = splits
            trailer_query = None

        output: Dict[str, torch.Tensor] = {}
        if self._config.bev_semantic_weight > 0:
            output["bev_semantic_map"] = self._bev_semantic_head(bev_feature_upscale)

        # Truck trajectory: 기존 키 "trajectory" 유지
        trajectory = self._trajectory_head(trajectory_query)
        output.update(trajectory)

        # Trailer trajectory: 토글이 켜져 있을 때만 forward · 출력. 출력 키가 없으면
        # loss.py / evaluate.py 둘 다 자동 skip (둘 다 키 존재 여부로 가드).
        if self._config.use_trailer_head:
            trailer_pred = self._trailer_trajectory_head(trailer_query)
            output["trailer_trajectory"] = trailer_pred["trajectory"]

        agents = self._agent_head(agents_query)
        output.update(agents)

        return output


class AgentHead(nn.Module):
    def __init__(self, num_agents: int, d_ffn: int, d_model: int):
        super().__init__()
        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp_states = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, BoundingBox2DIndex.size()),
        )
        self._mlp_label = nn.Sequential(
            nn.Linear(self._d_model, 1),
        )

    def forward(self, agent_queries) -> Dict[str, torch.Tensor]:
        agent_states = self._mlp_states(agent_queries)
        agent_states[..., BoundingBox2DIndex.POINT] = agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
        agent_states[..., BoundingBox2DIndex.HEADING] = agent_states[..., BoundingBox2DIndex.HEADING].tanh() * np.pi
        agent_labels = self._mlp_label(agent_queries).squeeze(dim=-1)
        return {"agent_states": agent_states, "agent_labels": agent_labels}


class TrajectoryHead(nn.Module):
    def __init__(self, num_poses: int, d_ffn: int, d_model: int):
        super().__init__()
        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, num_poses * StateSE2Index.size()),
        )

    def forward(self, object_queries) -> Dict[str, torch.Tensor]:
        poses = self._mlp(object_queries).reshape(-1, self._num_poses, StateSE2Index.size())
        poses[..., StateSE2Index.HEADING] = poses[..., StateSE2Index.HEADING].tanh() * np.pi
        return {"trajectory": poses}
