"""TransfuserConfig dataclass 정의.

이 파일은 **schema + v3 baseline default**를 담는다. 각 실험 버전은
`configs/vN_*.py`에서 `TransfuserConfig(...)` 호출 시 override만 명시하는
delta 패턴으로 관리. (full-snapshot 패턴은 diff가 noisy해서 채택 안 함)

여기 default 값은 trailer_v3 학습 시점 설정과 일치 (`docs/06-baseline-validation.md`
§1 ~ §5 참조). 따라서 `configs/v3_baseline.py`의 `TransfuserConfig()`는
override 없이 그대로 v3 재현이 가능.

base 자체를 수정하면 모든 후속 vN의 의미가 바뀌므로 신중. NavSim과 동일 항목은
NavSim 표준을 따르되, TruckScenes 적용 시 의도적으로 변경한 항목
(camera_width 1536, status feature 4D, trailer head, bev_semantic 0 등)은
`docs/04-current-port-review.md`에서 정당화 완료.
"""
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class TransfuserConfig:
    """TransFuser × TruckScenes 전체 hyperparameter."""

    # === Backbone (NavSim과 동일) ===
    image_architecture: str = "resnet34"
    lidar_architecture: str = "resnet34"

    latent: bool = False
    latent_rad_thresh: float = 4 * np.pi / 9

    # === LiDAR pre-processing ===
    max_height_lidar: float = 100.0     # ego frame z 상한 (사실상 무필터)
    pixels_per_meter: float = 4.0       # BEV 해상도 (NavSim과 동일)
    hist_max_per_pixel: int = 5         # histogram clip — pixel당 최대 점 수

    # BEV 범위. NavSim default ±32m 대칭 그대로 (trailer_v3 학습 시 사용된 값).
    # forward range가 부족하면 highway에서 곡선 진입을 못 봐 직진 prior로 회귀.
    # → 변경은 `configs/v4_range.py` 등 후속 버전에서 override.
    lidar_min_x: float = -32
    lidar_max_x: float = 32
    lidar_min_y: float = -32
    lidar_max_y: float = 32

    # below(<= split) / above(> split) 채널 분리 기준 (ego frame 지면 z=0).
    lidar_split_height: float = 0.2
    # True면 [below, above] 2채널, False면 above 1채널만. NavSim default False.
    use_ground_plane: bool = False

    lidar_seq_len: int = 1   # multi-frame LiDAR 미사용

    # === Image / BEV 입력 grid ===
    # camera_width 1536 = NavSim 1024 → TruckScenes 4-cam stitching에 맞춰 확장.
    camera_width: int = 1536
    camera_height: int = 256
    # ResNet ÷32 stem 가정 + fusion transformer pos_emb이 빌드 타임 고정 →
    # range가 바뀌어도 lidar_resolution_*가 256² 이면 모델 차원 변경 불요.
    # range × pixels_per_meter == lidar_resolution_*가 되도록 v? config에서 override.
    # (예: x range 80m × 4 px/m → lidar_resolution_height=320)
    # lidar_vert/horz_anchors도 함께 lidar_resolution_*/32로 맞춰야 GPT pos_emb 정합.
    lidar_resolution_width: int = 256
    lidar_resolution_height: int = 256

    img_vert_anchors: int = 256 // 32
    img_horz_anchors: int = 1536 // 32
    lidar_vert_anchors: int = 256 // 32
    lidar_horz_anchors: int = 256 // 32

    # === GPT fusion ===
    block_exp = 4
    n_layer = 2
    n_head = 4
    n_scale = 4
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    gpt_linear_layer_init_mean = 0.0
    gpt_linear_layer_init_std = 0.02
    gpt_layer_norm_init_weight = 1.0

    perspective_downsample_factor = 1
    transformer_decoder_join = True
    detect_boxes = True
    use_bev_semantic = False  # TruckScenes에 HD map 없어서 비활성
    use_semantic = False
    use_depth = False
    add_features = True

    # === Trajectory / agent decoder transformer ===
    tf_d_model: int = 256
    tf_d_ffn: int = 1024
    tf_num_layers: int = 3
    tf_num_head: int = 8
    tf_dropout: float = 0.0

    # === Detection (auxiliary) ===
    num_bounding_boxes: int = 30

    # === Loss weights ===
    trajectory_weight: float = 10.0
    agent_class_weight: float = 10.0
    agent_box_weight: float = 1.0
    bev_semantic_weight: float = 0.0  # disabled: no HD map
    # ego_trailer 미래 trajectory L1 loss 가중치.
    # truck trajectory와 동일한 10.0으로 설정하되, trailer 없는 sample은
    # mask=0이라 자동 제외되므로 batch별 학습 신호는 비례적으로 줄어듦.
    trailer_weight: float = 10.0

    # trailer trajectory head 자체를 모델에서 빼는 토글.
    # True (default): query slot + head 빌드, forward에서 trailer trajectory 예측·출력
    # False: query 수가 줄고 trailer head 빌드 안 함 → forward / loss / eval 모두 skip
    # paper의 "truck-only baseline"용. trailer_weight=0만 두면 forward는 그대로 돌아
    # capacity가 같지 않음 → strict ablation에는 이 flag로 완전히 제거해야 함.
    use_trailer_head: bool = True

    # === BEV semantic head (구조만 살아있음, weight=0) ===
    bev_pixel_width: int = 256
    bev_pixel_height: int = 128
    bev_pixel_size: float = 0.25

    num_bev_classes = 7
    bev_features_channels: int = 64
    bev_down_sample_factor: int = 4
    bev_upsample_factor: int = 2

    # === Trajectory target ===
    num_poses: int = 8                          # 8 future poses
    trajectory_sampling_time: float = 4.0       # 총 4초
    trajectory_sampling_interval: float = 0.5   # 0.5초 간격 (= keyframe interval)

    # Status feature dropout — 학습 중 batch sample의 일부를 status=0으로 마스킹.
    # vx/vy가 trajectory와 trivial mapping(vx·Δt)을 만들어 모델이 image/lidar를
    # 무시하는 문제 방지. 0.0=비활성, 0.5=50% sample을 0으로.
    # 근거: trailer_v1(chassis 30% fake-zero)이 trailer_v2(clean)보다
    # vision branch를 8배 더 많이 사용 → 의도적으로 50%로 끌어올려 vision 학습 강제.
    status_dropout_p: float = 0.5

    # NavSim TransFuser baseline parity: concat the 3-way one-hot driving_command
    # ([Turn Right, Turn Left, Go Straight]) into the status encoder input.
    # Default False so existing v3~v8 configs / checkpoints stay unaffected.
    use_driving_command: bool = False

    # VAD §4.2 protocol: drop ego status (vx, vy, ax, ay) from model input to
    # avoid shortcut learning in open-loop planning evaluation
    # ("VAD omits ego status features to avoid shortcut learning ... but the
    #  results of VAD using ego status features are still preserved in Tab. 1
    #  for reference"). Default True for v3~v7 compatibility — set False in
    #  configs that want VAD-style omission. When False the status encoder
    #  drops the (vx, vy, ax, ay) channels and status_dropout has no effect.
    use_ego_status: bool = True

    # How driving_command is derived from the future ego trajectory.
    # "heading": threshold |Δyaw@4s| (rad cast from
    #            `driving_command_threshold_heading_deg`) — closer to
    #            NavSim/nuPlan's route-derived intent (intersection turns
    #            only; lane changes stay STRAIGHT).
    # "lateral": threshold |local_y of last future pose| in meters
    #            (`driving_command_threshold_lateral_m`) — VAD's original
    #            nuScenes-converter pattern; conflates lane change with
    #            real turns. Kept for ablation.
    # Only consulted when use_driving_command=True.
    driving_command_mode: str = "heading"
    driving_command_threshold_heading_deg: float = 15.0
    driving_command_threshold_lateral_m: float = 2.0

    # === Optimizer / LR schedule ===
    # default는 v3·v4·v5 학습에 쓴 설정 (Adam, no weight_decay, no warmup).
    # NavSim 표준에 가까이 맞추려면 v6_lr_schedule처럼 AdamW + weight_decay + warmup.
    optimizer: str = "adam"            # "adam" | "adamw"
    weight_decay: float = 0.0          # AdamW일 때 의미 있음
    lr_warmup_epochs: int = 0          # 0이면 warmup 없이 바로 cosine annealing 시작

    # === Derived properties ===
    @property
    def bev_semantic_frame(self) -> Tuple[int, int]:
        return (self.bev_pixel_height, self.bev_pixel_width)

    @property
    def bev_radius(self) -> float:
        values = [
            self.lidar_min_x,
            self.lidar_max_x,
            self.lidar_min_y,
            self.lidar_max_y,
        ]
        return max([abs(value) for value in values])
