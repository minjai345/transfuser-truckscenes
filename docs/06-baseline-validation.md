# 06. Baseline 검증 — Constant-Velocity vs trailer_v3

> **상태**: 2026-04-29 측정 완료. trailer_v3 ep12 (status_dropout=0.5) 기준.
> **결론**: Aggregate L2로는 const-vel이 trailer_v3를 이김. 그러나 sample-level 곡률 분포가 84% 직진 dominant라서 평균이 misleading. **곡률 bin별로 분리하면 trailer_v3가 곡선/회전 sample에서 const-vel을 명확히 깸** → paper baseline 자격 유지.

---

## 1. 동기

trailer head 학습된 모델(trailer_v3 ep12)을 articulation-aware metric paper의 **baseline**으로 쓸 수 있는지 검증해야 함.

조건: 최소한 trivial baseline(constant velocity 외삽)보다는 나아야 paper에서 "non-trivial baseline"으로 통과.

측정 도구:
- `tools/checks/check_const_vel_baseline.py` — const-vel L2/collision 측정
- `tools/checks/check_dataset_curvature_dist.py` — sample-level 곡률 분포
- `tools/checks/check_stratified_eval.py` — bin별 분리 평가

---

## 2. 1차 측정: Aggregate L2 — trailer_v3가 짐

`val full = 2,395 samples` 대상.

| Method | Truck Avg L2 | Trailer Avg L2 | Col Avg |
|---|---|---|---|
| **Const-vel** | **0.66m** | **0.66m** | **0.39%** |
| trailer_v2 ep12 (status cheat) | 0.56m | — | — |
| **trailer_v3 ep12** (status_dropout=0.5) | 1.05m | 1.34m | 0.60% |

→ aggregate만 보면 **const-vel > trailer_v3**. Paper baseline으로 쓸 수 없어 보임.

이유 추측: TruckScenes는 highway 위주 → 직선 dominant → const-vel이 trivially 잘함.

---

## 3. 2차 측정: 데이터셋 곡률 분포 — 84% 직진

`tools/checks/check_dataset_curvature_dist.py` 결과 (val n=2,395).

### 3.1 Sample-level |Δheading @ 3s|

| bin | sample 수 | 비율 | const-vel L2 mean |
|---|---|---|---|
| `[0, 1°)` | 1,532 | **64.0%** | 0.62m |
| `[1, 3°)` | 488 | 20.4% | 1.39m |
| `[3, 5°)` | 127 | 5.3% | 2.18m |
| `[5, 10°)` | 113 | 4.7% | 2.72m |
| `[10, 20°)` | 52 | 2.2% | 5.13m |
| `[20, 90°)` | 83 | 3.5% | 5.71m |

- median Δh = **0.59°**, 90-percentile = 5.25° → 90% 샘플이 차선 유지 수준
- 측방 변위 median 0.30m, mean 0.75m → 거의 직선 이동
- median 속도 = 70 km/h, mean = 58 km/h → 고속도로 dominant

### 3.2 Trailer 곡률(articulation 효과)

with-trailer sample n=1,819 중:
- <3° (직진): 84%
- 3–10°: 11%
- ≥10° (강한 articulation): **5.6%**

→ articulation 효과는 데이터셋 전체의 ~5%에서만 나타남.

### 3.3 함의

- **Const-vel L2 0.66m는 64%의 직진 sample에 dominated** — 직선 sample만으로는 평균 0.34m
- **곡선 5.6% sample에서는 const-vel L2 5.49m** — 완전히 깨짐
- aggregate L2는 baseline 비교에 부적합. **bin별 분리 평가 필요**.

---

## 4. 3차 측정: Stratified Evaluation — trailer_v3가 곡선에서 const-vel 깸

`tools/checks/check_stratified_eval.py` 결과. 같은 val 2,395 sample을 `|Δheading @ 3s|`로 4 bin 분리.

### 4.1 Truck L2 (avg of 1s/2s/3s)

| bin | 비율 | const-vel | trailer_v3 ep12 | Δ | 승자 |
|---|---|---|---|---|---|
| straight (<1°) | 64.0% | 0.34m | 1.05m | +0.72m | const-vel (3× 더 좋음) |
| lane (1–3°) | 20.4% | 0.74m | 0.98m | +0.24m | const-vel (소폭) |
| moderate (3–10°) | 10.0% | 1.29m | **0.83m** | −0.46m | **trailer_v3** ✓ |
| **curvy (≥10°)** | 5.6% | **2.86m** | **1.72m** | **−1.14m** | **trailer_v3 ✓ (40% 향상)** |

### 4.2 Trailer L2 (avg of 1s/2s/3s)

| bin | const-vel | trailer_v3 ep12 | Δ |
|---|---|---|---|
| straight | 0.40m | 1.48m | +1.08m |
| lane | 0.70m | 1.12m | +0.42m |
| moderate | 1.06m | 0.97m | −0.09m |
| **curvy** | **2.01m** | **1.48m** | **−0.54m (27% 향상)** |

### 4.3 Per-horizon — curvy bin (가장 paper에 중요)

| horizon | metric | const-vel | trailer_v3 |
|---|---|---|---|
| 1s | truck | 0.62m | **0.47m** |
| 2s | truck | 2.47m | **1.48m** |
| 3s | truck | **5.49m** | **3.21m (-41%)** |
| 1s | trailer | 0.44m | 0.59m |
| 2s | trailer | 1.60m | **1.24m** |
| 3s | trailer | **4.00m** | **2.60m (-35%)** |

→ horizon이 길수록 격차가 더 커짐. const-vel은 **단순 외삽이라 곡선에서 시간 비례로 발산**. 학습 모델은 vision/lidar로 회전 의도를 잡아 발산을 억제.

---

## 5. Paper 시나리오 정당성

### 5.1 baseline 자격 ✓

trailer_v3는 **곡선/회전 시나리오에서 명확히 const-vel을 깸**. 직선 dominant에 가려져 있던 학습 효과가 stratified로 보면 드러남.

→ paper에서 trailer_v3를 baseline으로 사용 가능. 단 evaluation은 **bin별 / curvy-subset**으로 보고해야 함.

### 5.2 Aggregate 수치의 함정 — paper에서 명시

TruckScenes(및 유사 highway dataset)는 직진 dominant. aggregate L2는 baseline 차이를 묻어버림. **곡선 sample subset 또는 articulation-weighted metric**이 의미 있는 비교 수단.

→ paper의 한 sub-section으로 "Why aggregate L2 misleads on highway data" 들어갈 수 있음 (이 측정 자체가 contribution 일부).

### 5.3 Articulation 효과 sample 비율

with-trailer 곡선(≥10°) sample = **102개 (4.3% of val)**. 충분히 통계적 비교는 가능하나 sample 크기 작음 → 분산 큼.

→ paper에서 1-seed 결과만 제시하면 fragile. **2~3 seed 학습 후 mean ± std** 보여주는 게 안전.

---

## 6. 시각적 관찰: 모델이 곡선에서 직진 경향

수치는 const-vel을 깨지만, 시각화상 trailer_v3 ep12는 여전히:
- 곡선 시작 시 반응이 살짝 늦음 (open-loop L1 학습의 일반적 한계)
- 길이 보이면 GT 노선과 다른 길로 가려는 경향 (routing signal 부재)
- 강한 회전을 끝까지 따라가지 못하고 직진 prior로 회귀

이는 **input 부족(routing signal)**, **L1 averaging**, **camera/lidar feature가 회전 단서를 충분히 못 잡음**의 복합. 다음 절 참고.

---

## 7. 다음 단계 후보 (우선순위)

paper 진행 + baseline 추가 강화 두 갈래.

### 7.1 Paper main contribution (즉시 시작 가능)
- (a) **Articulation-aware metric 정의** — jackknife angle, trailer cut-in deviation, articulation residual L2
- (b) **Kinematic trailer head 설계** — truck pose + hitch angle → trailer trajectory (single-track 모델). free-regression baseline 대비 곡선에서 명확한 향상 기대

### 7.2 Baseline 강화 (paper 보강용, 우선순위 후순위)
- (c) **LiDAR 2채널** (`use_ground_plane=True`) — 6 LiDAR가 ego 주변을 dense하게 깔아주므로 below-ground 채널이 도로 형상 단서를 줄 가능성. 곡선에서 효과 클 것으로 추정
- (d) **Routing signal 추가** — driving_command (4-class one-hot) 또는 future heading delta. 갈림길에서 GT 의도 부재 문제 해결
- (참고) ImageNet 정규화는 NavSim·CARLA 모두 미적용이라 reference consistency 정책상 추가하지 않음 (`docs/README.md` 정책 행 참조)

### 7.3 검증·문서화 (paper writing 시 필요)
- (f) **곡률 bin별 multi-seed 학습** — 현재 1 seed로는 곡선 102 sample에서 noise 큼
- (g) **Articulation 효과가 강한 sub-scene 정량화** — `data/scene_curvy_split.json` top-N과 sample-level top-N을 paper용 표로

---

## 8. 측정 재현

```bash
# Const-vel baseline
python tools/checks/check_const_vel_baseline.py

# 곡률 분포
python tools/checks/check_dataset_curvature_dist.py

# Stratified eval (모델 ckpt 필요)
python tools/checks/check_stratified_eval.py \
    --checkpoint work_dirs/trailer_v3_status_dropout50_20260428_153540/checkpoints/epoch12.pt
```

세 스크립트 모두 deterministic — 같은 seed/같은 ckpt면 같은 결과 재현됨.
