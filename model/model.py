from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from model.backbone import TransfuserBackbone
from model.config import TransfuserConfig
from model.enums import BoundingBox2DIndex, StateSE2Index


class TransfuserModel(nn.Module):
    """Torch module for Transfuser."""

    def __init__(self, config: TransfuserConfig):
        super().__init__()

        # Query 구성: [truck trajectory(1), trailer trajectory(1), agents(N)]
        # truck/trailer를 별도 query로 분리해 각자 다른 head로 회귀.
        # trailer query는 현재 sample에 ego_trailer 없으면 loss에서 mask=0으로 제외됨.
        self._query_splits = [
            1,                          # truck trajectory query
            1,                          # trailer trajectory query
            config.num_bounding_boxes,  # agent detection queries
        ]

        self._config = config
        self._backbone = TransfuserBackbone(config)

        self._keyval_embedding = nn.Embedding(8**2 + 1, config.tf_d_model)
        self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)

        self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)
        self._status_encoding = nn.Linear(4, config.tf_d_model)

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
        status_feature: torch.Tensor = features["status_feature"]

        batch_size = status_feature.shape[0]

        bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature, lidar_feature)

        bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1)
        bev_feature = bev_feature.permute(0, 2, 1)
        status_encoding = self._status_encoding(status_feature)

        keyval = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1)
        keyval += self._keyval_embedding.weight[None, ...]

        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        query_out = self._tf_decoder(query, keyval)

        # query_splits 순서: [truck, trailer, agents]
        trajectory_query, trailer_query, agents_query = query_out.split(self._query_splits, dim=1)

        output: Dict[str, torch.Tensor] = {}
        if self._config.bev_semantic_weight > 0:
            output["bev_semantic_map"] = self._bev_semantic_head(bev_feature_upscale)

        # Truck trajectory: 기존 키 "trajectory" 유지
        trajectory = self._trajectory_head(trajectory_query)
        output.update(trajectory)

        # Trailer trajectory: TrajectoryHead가 {"trajectory": ...}를 반환하므로 키 변경
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
