"""v6 = v5_truck_only base + LR schedule을 NavSim 표준에 맞춤.

배경:
  v5_truck_only ep20 baseline (curvy 1.58, truck 1.00) — paper에서 reviewer가
  weak baseline으로 볼 가능성. NavSim 표준에 가까이 맞추는 정공법으로 정량 향상.

v3·v4·v5 학습 설정 (default):
  - Adam optimizer (no weight decay)
  - lr 1e-4 (constant 시작)
  - CosineAnnealingLR (T_max=20 epochs)
  - warmup 없음 → 학습 초기 unstable

이 v6에서 변경:
  - optimizer: Adam → AdamW (NavSim 표준)
  - weight_decay: 0 → 0.01 (NavSim 표준 regularization)
  - lr_warmup_epochs: 0 → 1 (linear warmup 1 epoch + 19 epoch cosine)

기대:
  - 학습 초기 안정화 → plateau 더 깊이 도달
  - weight_decay regularization → overfitting 약화 (특히 detection task)
  - 모든 metric 일관 향상 (직선·곡선·LiDAR Δ 모두)

baseline 적합성: NavSim 표준 element를 추가하는 거라 reviewer가 받아들일 수
있는 변경. paper에선 "표준 따라감"이라 부담 0.

base는 v5_truck_only (range_full + no trailer head).
"""
from configs._base import TransfuserConfig

config = TransfuserConfig(
    lidar_min_x=-32,
    lidar_max_x=48,
    lidar_resolution_height=320,
    lidar_vert_anchors=10,
    use_trailer_head=False,
    trailer_weight=0.0,
    optimizer="adamw",
    weight_decay=0.01,
    lr_warmup_epochs=1,
)
