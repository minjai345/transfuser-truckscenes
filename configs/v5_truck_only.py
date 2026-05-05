"""v5 truck-only ablation — v5_range_full에서 trailer head 제거.

용도: paper의 articulation metric / kinematic head contribution 비교군.
v5_range_full(현 best baseline)과 같은 BEV·grid 설정에서 trailer query slot · head
자체를 빌드 안 해서 capacity까지 truck-only.

변경 (delta from v5_range_full):
  - use_trailer_head: True → False
    → query_splits에서 trailer slot 빠짐 (`_query_splits = [1, num_bounding_boxes]`)
    → `_trailer_trajectory_head` 빌드 안 함
    → forward에서 trailer_trajectory 출력 안 함 → loss / evaluate 자동 skip
  - trailer_weight: 10.0 → 0.0 (use_trailer_head=False면 어차피 무시되지만 명시)

BEV range / grid는 v5_range_full 그대로 유지:
  - x: [-32, 48]m, y: [-32, 32]m
  - lidar_resolution_height=320, lidar_vert_anchors=10
"""
from configs._base import TransfuserConfig

config = TransfuserConfig(
    lidar_min_x=-32,
    lidar_max_x=48,
    lidar_resolution_height=320,
    lidar_vert_anchors=10,
    use_trailer_head=False,
    trailer_weight=0.0,
)
