"""v4 truck-only ablation — v4_range에서 trailer head 제거.

용도: paper의 articulation metric / kinematic head contribution 비교군.
trailer_weight=0만 두면 forward는 그대로라 capacity가 같지 않음.
이 변형은 query slot · head 자체를 빌드 안 해서 진짜 truck-only baseline.

변경 (delta from v4_range):
  - use_trailer_head: True → False
    → query_splits에서 trailer slot 빠짐 (`_query_splits = [1, num_bounding_boxes]`)
    → `_trailer_trajectory_head` 빌드 안 함
    → forward에서 trailer_trajectory 출력 안 함 → loss / evaluate 자동 skip
  - trailer_weight: 10.0 → 0.0
    → 명시적으로 supervision도 0 (use_trailer_head=False면 출력이 없어 자동 무시됨)

BEV range는 v4_range 그대로 유지 (lidar_min_x=-16, lidar_max_x=48).
"""
from configs._base import TransfuserConfig

config = TransfuserConfig(
    lidar_min_x=-16,
    lidar_max_x=48,
    use_trailer_head=False,
    trailer_weight=0.0,
)
