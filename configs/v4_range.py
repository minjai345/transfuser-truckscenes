"""v4 = v3 + forward-biased BEV range.

진단 (`docs/06-baseline-validation.md` §6, §7.2(c)):
  - TruckScenes는 highway dominant (median 70 km/h). 4초 horizon에 78m 전방까지 가야 함
  - v3의 BEV는 x ∈ [-32, 32]m → 전방 32m까지만 보여서 곡선이 시작도 안 보임
  - 모델은 입력에 보이지 않는 곡선을 예측 못 하고 직진 prior로 회귀

변경 (delta from v3):
  - lidar_min_x: -32  →  -16   (트레일러 ~15m cover, 1m 여유)
  - lidar_max_x:  32  →   48   (전방 48m, 4초 × 16 m/s 평균 cover)
  - y range, 해상도, ground_plane: v3와 동일

폭 64m 그대로 → grid 256×256 동일 → 모델 차원 변경 없음.
v3 ckpt와 호환은 안 됨 (LiDAR feature가 다른 spatial extent를 표현하므로 의미가 다름) →
from-scratch 학습.
"""
from configs._base import TransfuserConfig

config = TransfuserConfig(
    lidar_min_x=-16,
    lidar_max_x=48,
)
