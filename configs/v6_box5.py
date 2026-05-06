"""v6 = v5_truck_only base + agent_box_weight 1 → 5. LiDAR aux supervision 강화.

진단 (`docs/06-baseline-validation.md` 종합):
  - 시각화상 detection 결과가 거의 쓸 만하지 않음 (대부분 mismatch)
  - input ablation에서 LiDAR Δ +0.5m 정도로 modest. camera-dominant 양상
  - LiDAR-side에 붙은 supervised task는 agent_head 하나뿐인데 weight=1로 약함
    → LiDAR encoder가 trajectory loss의 carrier 역할만 하고 직접적 학습 신호 약함

처방: agent detection loss weight를 5배로 키워 LiDAR encoder가 detection task
로 더 강하게 학습되도록 강제. 부수효과로 detection 자체 품질도 개선 기대.

변경 (delta from v5_truck_only):
  - agent_box_weight: 1.0 → 5.0
  - agent_class_weight, trajectory_weight 등 다른 weight는 그대로

trade-off:
  - trajectory loss 비중이 상대적으로 줄어듦 → trajectory 학습 약화 가능성
  - agent_class_weight=10이 그대로라 이미 분류는 강한 신호. box regression만 강화
  - 정량 L2가 약간 손해볼 수 있지만 LiDAR Δ가 커지면 paper에선 trade-off로 보고

검증 포인트:
  - LiDAR Δ (no_lidar effect) 증가하는지 — 핵심 KPI
  - detection 시각화에서 차량 박스가 더 잘 잡히는지
  - truck L2가 과하게 손해보지 않는지 (>0.2m 손해면 너무 비쌈)
"""
from configs._base import TransfuserConfig

config = TransfuserConfig(
    lidar_min_x=-32,
    lidar_max_x=48,
    lidar_resolution_height=320,
    lidar_vert_anchors=10,
    use_trailer_head=False,
    trailer_weight=0.0,
    agent_box_weight=5.0,
)
