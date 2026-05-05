"""v5 = v4_range에 backward 회복 추가. 곡선 인지 fix용.

진단 (`docs/06-baseline-validation.md` §8):
  v4_range는 직선 sample(84%)에서 향상. 곡선(15%)에서는 v3 ep12보다 후퇴
  (truck +0.57m, trailer +0.33m at curvy bin).

원인 가설:
  v4의 후방 −16m → 트레일러(ego 뒤쪽 ~−15m까지 뻗음)가 BEV 모서리에 1m 여유로
  걸침. 곡선에서 트럭이 회전하면 트레일러가 swing → 그 점들이 BEV 외곽에서
  잘려나감 → 자기 자신의 회전 신호가 input에서 사라짐 → 모델이 직진 prior로 회귀.
  직선에서는 trailer가 swing 안 해서 잘려도 영향 없음 → forward 확장 이득만 보임.

처방: forward 48m(v4 그대로) + backward 32m(v3 수준) → 둘 다 cover.
  - x: [-32, 48]m  (80m 폭)
  - y: [-32, 32]m  (64m, v4 동일)
  - 4 px/m → grid 320×256 (v3·v4의 256²에서 변경)

영향:
  - lidar_resolution_height: 256 → 320 (x축 grid)
  - lidar_vert_anchors: 8 → 10 (=320/32)  → GPT pos_emb 변경 → from-scratch
  - GPT cross-attention cost ~N²: tokens 64→80 (1.4× 비용, 학습 시간 ~3h 증가)
  - 메모리: BEV histogram 1.25×, ckpt 호환성 없음 (pos_emb shape 다름)
"""
from configs._base import TransfuserConfig

config = TransfuserConfig(
    lidar_min_x=-32,
    lidar_max_x=48,
    lidar_resolution_height=320,
    lidar_vert_anchors=10,
)
