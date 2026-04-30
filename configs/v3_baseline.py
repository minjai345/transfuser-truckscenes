"""v3 baseline — `_base.py` default 그대로.

이 버전이 trailer_v3_status_dropout50 학습에 사용된 설정.
검증 결과는 `docs/06-baseline-validation.md`.

핵심 사양:
  - BEV 범위 ±32m 대칭 (NavSim default)
  - LiDAR 1채널 (above split_height=0.2m만)
  - status feature 4D (vx/vy from ego_pose 미분 + chassis ax/ay)
  - status_dropout_p=0.5 (vision 학습 강제)
  - trailer head + trailer_weight=10.0

→ override 없음. base 자체가 v3 default.
"""
from configs._base import TransfuserConfig

config = TransfuserConfig()
