"""v5_truck_only + status_dropout 제거. dropout 정책 검증.

진단:
  v3 도입 시 status_dropout_p=0.5로 vx/vy의 trivial mapping(vx·Δt) 막으려 했음.
  근거: v2(chassis 32% fake-zero) → v3(clean derived + dropout 0.5)에서 정량 0.5m
  악화. 단 v2→v3는 input source까지 같이 바뀐 두 변수 동시 변경이라 dropout
  자체가 원인이라고 단정 어려움.

이 변형이 검증할 것:
  - clean status + dropout 0이 v5_truck_only(clean status + dropout 0.5)보다
    truck L2가 좋아지면 → dropout이 정량을 깎고 있던 것 (사용자 직감 맞음)
  - input ablation `no_status` 효과를 같이 봐야 함:
    - 효과 견딜만하면 → dropout 없이도 vision/lidar 학습 잘 됨. 정책 변경
    - 효과 폭발적이면 → status에 trivial 의존. 정량 좋아도 paper baseline 부적합

trailer GT annotation noise 이슈로 v5_truck_only base 채택 (trailer head 안 쓰고
truck planning만 평가).

변경 (delta from v5_truck_only):
  - status_dropout_p: 0.5 → 0.0

BEV·grid·trailer 설정은 v5_truck_only 그대로:
  - x: [-32, 48]m, y: [-32, 32]m, grid 320×256
  - use_trailer_head=False, trailer_weight=0.0
"""
from configs._base import TransfuserConfig

config = TransfuserConfig(
    lidar_min_x=-32,
    lidar_max_x=48,
    lidar_resolution_height=320,
    lidar_vert_anchors=10,
    use_trailer_head=False,
    trailer_weight=0.0,
    status_dropout_p=0.0,
)
