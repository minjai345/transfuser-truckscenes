"""v7 = v5_truck_only base + use_ground_plane=True. LiDAR 2채널 BEV.

배경:
  v5_truck_only baseline (curvy 1.58)에서 곡선 인지 강화 lever 중 시도 안 한
  유일한 표준 reference lever. CARLA TransFuser는 항상 use_ground_plane=True
  (above + below 2채널). NavSim default는 False지만 config flag로 지원.
  baseline에서 안 시도하면 reviewer가 "왜 빠뜨렸냐" 묻기 쉬움.

LiDAR feature 변경:
  - v5: 1채널 (above split_height=0.2m만) — 장애물 layer
  - v7: 2채널 [below, above] — 아래 채널은 도로 표면 layer

기대 효과:
  - 도로 형상(curvature, edges) 단서가 BEV input에 직접 표현
  - 곡선에서 도로 휘어짐을 모델이 input에서 직접 봐서 회전 prediction 강화
  - LiDAR Δ ↑ 가능 (LiDAR가 더 informative해짐)

base는 v5_truck_only (range_full + no trailer head + status_dropout 0.5).
optimizer는 default Adam (v6_lr_schedule는 효과 없었음).
"""
from configs._base import TransfuserConfig

config = TransfuserConfig(
    lidar_min_x=-32,
    lidar_max_x=48,
    lidar_resolution_height=320,
    lidar_vert_anchors=10,
    use_trailer_head=False,
    trailer_weight=0.0,
    use_ground_plane=True,
)
