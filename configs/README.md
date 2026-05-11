# Configs

실험 버전별 TransfuserConfig 관리 폴더.

## 구조

- **`_base.py`** — `TransfuserConfig` dataclass schema + v3 baseline default 값.
- **`vN_*.py`** — 각 실험 버전. `TransfuserConfig(...)` 호출 시 override만 명시.
- **`__init__.py`** — `load_config(name)` 헬퍼.

## 패턴

각 vN.py는 두 줄짜리 override delta + docstring으로 진단/근거를 남긴다:

```python
"""v4 = v3 + forward-biased BEV range. (진단/근거)"""
from configs._base import TransfuserConfig
config = TransfuserConfig(lidar_min_x=-16, lidar_max_x=48)
```

`diff configs/v3_baseline.py configs/v4_range.py`로 변경된 인자가 한눈에 보이게 하기 위함.

## 실행에 적용하는 법

학습 / 평가 / 시각화 스크립트의 `--config` flag로 버전 선택:

```bash
./scripts/train.py --config v7_ground_plane
python evaluate.py --config v7_ground_plane --checkpoint <path>
python tools/visualize.py --config v7_ground_plane ...
```

default는 paper baseline 버전 (현재 `v7_ground_plane` — 근거는 `docs/06-baseline-validation.md` §13).

## 버전 이력

| 버전 | 변경점 (delta from previous) | 동기 |
|---|---|---|
| `v3_baseline` | (no override — base 그대로) | NavSim 표준에 trailer head + status_dropout 0.5 추가. trailer_v3 학습에 사용 |
| `v4_range` | `lidar_min_x=-16, lidar_max_x=48` | TruckScenes highway에서 4s horizon 78m가 BEV 32m을 초과 → 곡선 인지 불가. 전방 48m으로 확장 |
| `v4_truck_only` | `+ use_trailer_head=False, trailer_weight=0.0` | paper의 strict truck-only baseline. trailer query slot · head 자체 빌드 skip → capacity까지 truck-only. v4_range와 같은 BEV 설정에서 trailer 유무 ablation |
| `v5_range_full` | `lidar_min_x=-32` (v4 −16에서 회복), `lidar_resolution_height=320`, `lidar_vert_anchors=10` | v4 곡선 후퇴 진단 fix: 후방 16m → 32m으로 회복(트레일러 swing cover) + forward 48m 유지. grid 320×256 → pos_emb shape 변경, from-scratch 학습 |
| `v5_truck_only` | `+ use_trailer_head=False, trailer_weight=0.0` | v5_range_full의 capacity-matched truck-only ablation. paper의 trailer head contribution 비교군 |
| `v5_truck_only_no_status_dropout` | `+ status_dropout_p=0.0` (v5_truck_only 기반) | dropout 정책 검증. v5_truck_only(dropout 0.5) vs 본 변형(dropout 0)으로 status_dropout이 정량 L2를 깎고 있는지 1:1 비교. 결과는 `docs/06 §9` |
| `v6_box5` | `+ agent_box_weight=5.0` (v5_truck_only 기반, loss weight 카테고리 변경이라 v6) | LiDAR aux supervision 강화. detection 품질 개선 + LiDAR encoder 학습 신호 ↑. paper의 LiDAR contribution 증가 검증. **결과**: 의도와 정반대 (LiDAR Δ 오히려 감소, curvy 후퇴). 조기 중단. `docs/06 §11` |
| `v6_lr_schedule` | `+ optimizer=adamw, weight_decay=0.01, lr_warmup_epochs=1` (v5_truck_only 기반) | NavSim 표준에 가까운 학습 schedule. 학습 안정화 + regularization으로 baseline 정량 향상 시도. **결과**: 효과 미미 (`docs/06 §13.3`) |
| `v7_ground_plane` | `+ use_ground_plane=True` (v5_truck_only 기반) | LiDAR 2채널 BEV (above + below split). paper TransFuser §3.2 표준 lever — 우리가 빠뜨렸던 것. **paper baseline 권장** (`docs/06 §13`) |
| `v8_range_back48` | `+ lidar_min_x=-48, lidar_resolution_height=384, lidar_vert_anchors=12` (v7_ground_plane 기반) | backward range -32 → -48m 추가 확장. paper 표준 ±32m에서 추가 일탈. hypothesis 약함 (단순 ablation) |

## 주의

- `_base.py`를 수정하면 **모든 vN의 default가 함께 바뀐다.** schema 추가/변경 시 vN의 의미 정합성 확인.
- ckpt는 vN 사양에 묶임. 다른 v로 resume할 때 호환 여부 직접 판단 필요 (입력 shape이나 channel이 바뀌면 첫 conv·pos_emb 재학습 필요).
- 새 vN을 만들 때 정수 sequence 이름(예: `v7_ground_plane`, `v8_range_back48`)으로 짓고, docstring에 직전 버전과의 delta + 근거 docs 링크를 명시.
