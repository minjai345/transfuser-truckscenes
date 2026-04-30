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
./scripts/train.py --config v4_range
python evaluate.py --config v4_range --checkpoint <path>
python tools/visualize.py --config v4_range ...
```

default는 최신 stable 버전 (현재 `v4_range`).

## 버전 이력

| 버전 | 변경점 (delta from previous) | 동기 |
|---|---|---|
| `v3_baseline` | (no override — base 그대로) | NavSim 표준에 trailer head + status_dropout 0.5 추가. trailer_v3 학습에 사용 |
| `v4_range` | `lidar_min_x=-16, lidar_max_x=48` | TruckScenes highway에서 4s horizon 78m가 BEV 32m을 초과 → 곡선 인지 불가. 전방 48m으로 확장 |
| `v4_truck_only` | `+ use_trailer_head=False, trailer_weight=0.0` | paper의 strict truck-only baseline. trailer query slot · head 자체 빌드 skip → capacity까지 truck-only. v4_range와 같은 BEV 설정에서 trailer 유무 ablation |

## 주의

- `_base.py`를 수정하면 **모든 vN의 default가 함께 바뀐다.** schema 추가/변경 시 vN의 의미 정합성 확인.
- ckpt는 vN 사양에 묶임. 다른 v로 resume할 때 호환 여부 직접 판단 필요 (입력 shape이나 channel이 바뀌면 첫 conv·pos_emb 재학습 필요).
- 새 vN을 만들 때 정수 sequence 이름(`v4_range`, `v5_xxx`)으로 짓고, docstring에 직전 버전과의 delta + 근거 docs 링크를 명시.
