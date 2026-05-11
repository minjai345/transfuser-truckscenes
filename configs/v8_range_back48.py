"""v8 = v7_ground_plane + backward range −32 → −48m.

배경:
  v7_ground_plane이 paper baseline 후보 (paper §3.2 ground_plane 정합 +
  LiDAR utilization 가장 강력). BEV는 forward 48m / backward 32m로 paper
  표준(±32m)에서 forward만 확장된 상태. backward도 +16m 확장해서 시야 더
  넓혀보는 ablation lever.

paper 충실성:
  - PAMI TransFuser §3.2: forward 32m, side ±16m. backward 명시 안 됨 (32m로 추정)
  - NavSim transfuser_config.py: ±32m 대칭
  → **둘 다 backward 32m이 표준**. v8 = paper 표준에서 추가 일탈

Hypothesis (효과 가능성):
  - 가설 (a) 트레일러 swing 인지 강화: 약함 (trailer 15m이라 32m로 cover 충분)
  - 가설 (b) 후방 차량 인지: 약함 (open-loop trajectory 결정에 영향 작음)
  - 가설 (c) 곡선에서 후방 단서: 약함 (forward 단서가 dominant)
  → 단순 ablation. 효과 없을 가능성 큼. 단 시도 가치는 있음 (paper에 보고)

변경 (delta from v7_ground_plane):
  - lidar_min_x: -32 → -48 (backward +16m)
  - lidar_resolution_height: 320 → 384 (= 96m × 4 px/m)
  - lidar_vert_anchors: 10 → 12 (= 384 / 32)

영향:
  - grid 384×256 → pos_emb shape 변경 → from-scratch 학습 (v7 ckpt 호환 X)
  - GPT cross-attention cost ~N²: tokens 80 → 96 (1.4× 비용, 학습 시간 ~3-5h 증가)
  - 메모리: BEV histogram 1.2x
"""
from configs._base import TransfuserConfig

config = TransfuserConfig(
    lidar_min_x=-48,
    lidar_max_x=48,
    lidar_resolution_height=384,
    lidar_vert_anchors=12,
    use_trailer_head=False,
    trailer_weight=0.0,
    use_ground_plane=True,
)
