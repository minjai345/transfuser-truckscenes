# 06. Baseline 검증 — Constant-Velocity vs trailer_v3 / v4

> **상태**:
> - 2026-04-29: trailer_v3 ep12 (status_dropout=0.5) 측정 완료 (§1-7).
> - 2026-05-01: trailer_v4 ep8 (BEV range -16~48m) 측정 완료 (§8-10).
>
> **결론**:
> - Aggregate L2로는 const-vel이 학습 모델을 이김. 그러나 sample-level 곡률 분포가 84% 직진 dominant라서 평균이 misleading. **곡률 bin별로 분리하면 학습 모델이 곡선/회전 sample에서 const-vel을 명확히 깸** → paper baseline 자격 유지.
> - v4 range 확장은 **직선·차로(84%)에서 v3보다 향상** but **곡선(15%)에서는 후퇴**. range 효과는 비대칭. 곡선 향상은 다른 lever 필요 (LiDAR ground plane, kinematic head 등).

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

## 8. v4_range 후속 검증 (2026-05-01)

> §7.2(c) 권고였던 LiDAR 2채널 대신, **forward-biased BEV range 확장**(v3 32m → v4 48m)을 먼저 시도. v4 학습 완료 후 v3 ep12와 직접 비교.

### 8.1 학습 결과 (val full 2,395 samples)

`configs/v4_range.py` (v3 default + `lidar_min_x=-16, lidar_max_x=48`)로 20 epoch 학습.

| 모델 | ckpt | Truck Avg L2 | Trailer Avg L2 | Col Avg |
|---|---|---|---|---|
| v3 ep12 (`trailer_v3_status_dropout50`) | epoch12.pt | 1.05m | 1.34m | 0.60% |
| **v4 ep8** (`trailer_v4_forward48`) | **epoch8.pt** | **0.98m** | **1.18m** | **0.43%** |
| v4 plateau (ep15-20 평균) | — | ~1.03m | ~1.33m | ~0.85% |

→ v4 ep8이 aggregate에서 v3 ep12를 모든 metric에서 갱신. 단 ep15-20 plateau는 v3 ep12와 비슷 수준 → ep8이 dip(단발성)일 가능성 있음.

### 8.2 v4 ep8 vs const-vel — stratified

같은 `tools/checks/check_stratified_eval.py` (val 2,395, --config v4_range로 호출).

#### Truck L2 (m, avg of 1/2/3s)

| bin | sample% | const-vel | v4_ep8 | Δ | 승자 |
|---|---|---|---|---|---|
| straight (<1°) | 64.0% | 0.34 | 0.90 | +0.57 | const-vel |
| lane (1-3°) | 20.4% | 0.74 | 0.87 | +0.13 | const-vel (소폭) |
| moderate (3-10°) | 10.0% | 1.29 | **0.96** | **−0.33** | **v4_ep8** ✓ |
| curvy (≥10°) | 5.6% | 2.86 | **2.29** | **−0.57** | **v4_ep8** ✓ (20% 향상) |

#### Trailer L2 (m, with-trailer sample만)

| bin | const-vel | v4_ep8 | Δ |
|---|---|---|---|
| straight | 0.40 | 1.22 | +0.82 |
| lane | 0.70 | 0.94 | +0.24 |
| moderate | 1.06 | 0.96 | −0.09 ≈ |
| curvy | 2.01 | **1.81** | **−0.20** ✓ |

→ moderate·curvy bin에서 const-vel을 일관되게 깸. **paper baseline 자격 유지**.

### 8.3 v4 ep8 vs v3 ep12 — bin별 직접 비교

#### Truck L2

| bin | v3_ep12 | v4_ep8 | Δ (v4−v3) | 해석 |
|---|---|---|---|---|
| straight | 1.05 | **0.90** | **−0.15** | v4 향상 |
| lane | 0.98 | **0.87** | **−0.11** | v4 향상 |
| moderate | **0.83** | 0.96 | +0.13 | v3 향상 |
| curvy | **1.72** | 2.29 | +0.57 | v3 향상 |

#### Trailer L2

| bin | v3_ep12 | v4_ep8 | Δ |
|---|---|---|---|
| straight | 1.48 | **1.22** | **−0.26** |
| lane | 1.12 | **0.94** | **−0.18** |
| moderate | **0.97** | 0.96 | −0.01 ≈ |
| curvy | **1.48** | 1.81 | +0.33 |

### 8.4 해석 — range 효과의 비대칭

- **직선·차로 (sample 84%)에서 v4가 일관되게 향상** (truck −0.11~−0.15, trailer −0.18~−0.26m). forward-biased range가 **장거리 직선 plan**에 유리. 멀리까지 BEV가 보이니 차로 따라가는 단서가 늘어남.
- **곡선 (15%)에서는 v4가 v3보다 후퇴** (truck +0.13~+0.57, trailer +0.33). 가능 원인:
  1. 후방을 32→16m으로 줄여 트레일러 swing 단서가 약해짐
  2. 곡선 sample이 적어 (135) 1-seed 분산 클 가능성
  3. ep8 단발성 dip — ep15-20 plateau ckpt에서는 곡선 회복 가능성 있음 (별도 측정 필요)
- **v3 의도("range 확장이 곡선 인지 도움")는 검증 실패**. 곡선 인지 향상은 다른 lever(LiDAR ground plane, multi-frame, kinematic trailer head 등) 쪽으로 가야 함.

### 8.5 LiDAR 의존도 추세 (학습 전반)

학습 중 input ablation (200 sample subset, 매 epoch):

| ep | full truck | no_lidar | LiDAR Δ | no_camera |
|---|---|---|---|---|
| 1-7 | 1.0-1.7 | 비슷 | ±0.2m (noise) | 4-6m (huge) |
| 8-20 plateau | ~1.0 | ~1.5 | **+0.4-0.7m** (consistent) | ~2m |

→ 학습 후반에 LiDAR 의존도가 **noise 수준에서 +0.5m로 증가**. range 확장이 LiDAR feature를 informative하게 만든 효과 일부 있음. 단 camera-dominant 양상은 유지.

### 8.6 Curvy scene 시각적 관찰

`scripts/visualize_curvy.py --ckpt v4_ep8 --top 3`로 cum_yaw 상위 3 scene 비디오 생성:
- idx 12 (cum_yaw 207°)
- idx 22 (cum_yaw 183°)
- idx 46 (cum_yaw 113°)

→ `viz/curvy_v4_ep8/`. v3과 동일 scene 비교 가능 (`viz/curvy/` 참고).

---

## 9. status_dropout 정책 검증 (2026-05-05, 조기 중단)

> **상태**: trailer_v5_truck_only_no_status_dropout (status_dropout_p=0) 학습.
> ep3 시점에서 패턴 명확 → 정량 절약 위해 중단.
> **결론**: status_dropout=0.5 유지가 정답. 빼면 trivial mapping(`vx·Δt`)으로 회귀.

### 9.1 배경

v2(chassis fake-zero) → v3(clean derived + dropout 0.5) 이행 시 정량 0.5m 악화
가 dropout 자체 때문인지 input source 변경 때문인지 단정 어려웠음. clean status
+ dropout 0 ablation으로 진짜 원인 분리 시도.

### 9.2 결과 — 부분적 정량 향상 + status 100% 의존

`configs/v5_truck_only_no_status_dropout.py` (= v5_truck_only + status_dropout_p=0):

| ep | truck Avg | straight | curvy | no_lidar | **no_status** |
|---|---|---|---|---|---|
| 1 | 0.95 | 0.65 | 3.03 | 1.17 | **32.58** ⚠️ |
| 2 | **0.88** | 0.66 | 2.65 | 1.05 | **33.00** ⚠️ |
| 3 | 0.99 | 0.67 | 2.83 | 0.96 | **33.07** ⚠️ |

비교 (v5_truck_only with dropout 0.5, ep20 plateau): truck 1.00 / straight 1.00 /
curvy 1.58 / no_status ~3-4m.

### 9.3 해석 — paper baseline으로 부적합

- ✓ **부분적 정량 향상**: truck full Avg 0.88 (vs 1.00), straight 0.67 (vs 1.00).
  사용자 직감 일부 맞음.
- ⚠️ **no_status ablation 32-33m**: status 입력 빼면 trajectory L2가 33m로 폭발 →
  **모델이 거의 100% status로 trivial mapping(`vx·Δt`)에 회귀**.
- ⚠️ **곡선 bin은 오히려 worse** (curvy 2.65-3.03 vs v5_truck_only의 1.58):
  trivial mapping은 곡선에서 무력. 회전 sample에서 모델이 vision/lidar 못 씀.

→ 직진 dominant 데이터셋(84%)에서 평균 L2가 좋아 보이는 건 trivial solution이
직진을 잘 평균낸 것일 뿐. **paper baseline으로는 부적합**.

### 9.4 정책 결정

- **status_dropout_p=0.5 유지** (v3 도입 그대로). v3·v4·v5 baseline 모두 이 설정.
- 사용자 직감 검증 의미는 있었음 — paper에 "naive `vx·Δt` regression이 평균 L2엔
  좋아 보이지만 status modality 제거 시 catastrophic failure" 정량 증거로 인용 가능.
- v3·v4·v5의 정량 결과는 status_dropout_p=0.5 기준이라 fair.

### 9.5 종료

ep3 결과로 결론 명확 → ep4-20 학습 미진행 (GPU 22h 절약).
ckpt: `work_dirs/trailer_v5_truck_only_no_status_dropout_20260505_180905/checkpoints/epoch{1,2,3}.pt`.

---

## 10. 다음 우선순위

§7 권고와 8.4 진단을 종합:

- **paper main contribution은 §7.1 그대로** (articulation metric + kinematic head). v3·v4 비교는 baseline 강화 측면에서 의미 있지만 paper main message는 아님.
- **곡선 인지 강화 lever**:
  - (i) **`use_ground_plane=True`** — §7.2(c). LiDAR 2채널이 도로 형상 단서 추가. 곡선 향상 기대.
  - (ii) **camera_dropout** — 학습 시 camera input 일부 sample 0으로 마스킹 → LiDAR encoder 학습 강제. v3·v4 모두 LiDAR 의존도 약함이 진단됨.
- **v4_truck_only ablation** — `configs/v4_truck_only.py` (use_trailer_head=False). v4_range와 같은 BEV에서 trailer head capacity 영향 측정. paper의 "trailer head 효과" 주장 정량화.
- **plateau ckpt 재측정** — ep15-20 중 한 ckpt를 stratified로 한 번 더 돌려서 ep8이 outlier인지 확인.

---

## 11. agent_box_weight 강화 검증 (2026-05-06, 조기 중단)

> **상태**: trailer_v6_box5 (agent_box_weight 1 → 5) 학습. ep10 시점에서 패턴
> 명확 → 정량 절약 위해 중단.
> **결론**: agent_box_weight up은 LiDAR encoder를 detection task에 가두어버려서
> 의도와 정반대 효과. LiDAR Δ가 오히려 감소. v5_truck_only가 여전히 best baseline.

### 11.1 배경

v3·v4·v5 모든 baseline에서 input ablation의 LiDAR Δ가 +0.5m 정도로 modest.
LiDAR encoder 학습 신호가 약한 게 원인. agent_head loss weight를 1 → 5로 키워
LiDAR-side aux supervision을 강화해 LiDAR encoder가 더 학습되도록 유도 시도.
부수효과로 detection 품질 개선도 기대 (시각화상 detection이 거의 못 잡음).

### 11.2 결과 — 의도와 정반대

`configs/v6_box5.py` (= v5_truck_only + agent_box_weight=5.0):

| ep | truck full | straight | curvy | LiDAR Δ |
|---|---|---|---|---|
| 5 | 0.94 | **0.79** | 2.68 | +0.04 |
| 6 | 1.02 | 0.84 | 2.60 | +0.32 |
| 7 | 0.94 | 0.87 | 2.66 | +0.08 |
| 8 | 1.01 | 0.97 | 2.63 | +0.10 |
| 9 | 0.93 | 0.89 | 2.51 | +0.18 |

비교 (v5_truck_only ep20 plateau): truck 1.00 / straight 1.00 / curvy 1.58 /
LiDAR Δ +0.54.

### 11.3 해석

- ✓ **직선·차로 향상**: straight 0.79~0.97 (vs v5의 1.00). detection 학습이
  spatial grounding에 도움된 부수효과 가능성.
- ⚠️ **곡선 후퇴**: curvy 2.51~2.68 (vs v5의 1.58). agent_box_weight up이
  trajectory loss 비중을 상대적으로 줄여 곡선 학습이 약해짐.
- ⚠️ **LiDAR Δ 오히려 감소**: +0.04~+0.32 (평균 ~+0.14, vs v5의 +0.54). 의도와
  정반대. agent_box_weight up이 LiDAR encoder를 **detection task에 가두어**
  trajectory에 흐르는 LiDAR 기여도는 오히려 줄어들었다는 해석. detection-specific
  feature와 planning-friendly feature가 같은 encoder에서 trade-off 관계.

→ "LiDAR aux supervision을 키우면 LiDAR가 trajectory에도 더 도움된다"는 가설
**기각**. encoder가 task-specific으로 분화되어 detection만 좋아지고 trajectory
에는 영향 적음. LiDAR encoder를 trajectory에 더 쓰게 하려면 다른 lever 필요
(camera_dropout, multi-frame status, 또는 BEV semantic supervision 부활).

### 11.4 종료

ep10 시점 plateau 도달, ep11-20 진행 안 함 (~14h GPU 절약).
ckpt: `work_dirs/trailer_v6_box5_20260506_001548/checkpoints/epoch{1..10}.pt`.

---

## 12. NavSim Standard 대비 각 config 일탈 (paper 충실성 점검)

> 2026-05-08 정리. NavSim TransFuser 코드 (`navsim/agents/transfuser/transfuser_agent.py`,
> `transfuser_config.py`) 직접 확인 후 각 버전이 어떻게 일탈했는지 명시.
> paper writing 시 "baseline은 NavSim default + 다음 dataset 차이만 적용"으로 disclose 가능.

### 12.1 NavSim default (paper strict standard)

코드 직접 인용:
```python
# transfuser_agent.py
torch.optim.Adam(model.parameters(), lr=self._lr)   # weight_decay 없음
TrajectorySampling(time_horizon=4, interval_length=0.5)   # T=8
# LR scheduler 없음 (constant lr)
# augmentation 없음
# status_dropout 없음

# transfuser_config.py
lidar_min_x/max_x/y = ±32          # ±32m 대칭
use_ground_plane = False           # 1ch BEV (above split_height만)
trajectory_weight = 10.0
agent_class_weight = 10.0
agent_box_weight = 1.0
bev_semantic_weight = 10.0         # HD map 사용
backbone = "resnet34"
# status: navigation_goal + velocity + acceleration (~8D, driving_command 포함)
```

### 12.2 데이터셋 차이로 인한 공통 일탈 (모든 v config 공유)

TruckScenes vs NuPlan(NavSim) 차이로 발생, **모든 v config 공통**:

| 항목 | NavSim | 우리 모든 v | 정당화 |
|---|---|---|---|
| camera | 3-cam, 1024×256 | 4-cam, 1536×256 | TruckScenes 4-cam config (LF/RF/LB/RB) |
| LiDAR | 1 cloud | 6 LiDAR ego frame 합성 | TruckScenes 6 LiDAR config |
| status 차원 | 8D (driving_cmd + vel + acc) | 4D (vel + acc) | TruckScenes에 driving_command raw 없음 |
| `bev_semantic_weight` | 10.0 | **0.0** | HD map 없음 → supervision 못 만듦 |
| trailer head | 없음 | `_trailer_trajectory_head` 추가 | articulated truck 적응 (paper main contribution scope) |
| `status_dropout_p` | 0 (없음) | **0.5** | clean status가 trivial mapping(`vx·Δt`) 회귀 유발 → vision/lidar 학습 강제 (검증: §9) |
| LR schedule | constant | `CosineAnnealingLR(T_max=epochs)` | 표준 transformer 학습 trick. NavSim 미적용이지만 일반 practice |

→ 위 7개 항목은 **TruckScenes 적응 + 일반 transformer practice**로 정당화. paper에서 "TruckScenes domain adaptation"으로 disclose.

### 12.3 각 v config의 추가 일탈 (NavSim default + 위 공통 일탈 외)

| config | 추가 일탈 (delta from "NavSim default + §12.2 공통") | 동기 |
|---|---|---|
| `v3_baseline` | 없음 (공통 일탈만) | NavSim에 가장 가까움. paper strict baseline |
| `v4_range` | `lidar_min_x=-16, lidar_max_x=48` | TruckScenes highway dominant, 4s × 평균속도 ≈ 78m → forward 32m 부족. 단 후방 -16m 짧음 |
| `v4_truck_only` | v4_range + `use_trailer_head=False, trailer_weight=0.0` | trailer head ablation (capacity 동일) |
| `v5_range_full` | `lidar_min_x=-32, lidar_max_x=48, lidar_resolution_height=320, lidar_vert_anchors=10` | v4 곡선 후퇴 fix: 후방 32m 회복 + forward 48m 유지. grid 320×256 |
| `v5_truck_only` | v5_range_full + `use_trailer_head=False, trailer_weight=0.0` | best baseline 후보. trailer head 제거 |
| `v5_truck_only_no_status_dropout` | v5_truck_only + `status_dropout_p=0.0` | status_dropout 정책 검증 (§9, 조기 중단 확인) |
| `v6_box5` | v5_truck_only + `agent_box_weight=5.0` | LiDAR aux 강화 시도 (§11, 효과 없음) |
| `v6_lr_schedule` | v5_truck_only + `optimizer="adamw", weight_decay=0.01, lr_warmup_epochs=1` | paper TransFuser §4.7의 AdamW + weight_decay 0.01 표준. NavSim은 Adam이라 일탈 |
| `v7_ground_plane` | v5_truck_only + `use_ground_plane=True` | paper TransFuser §3.2 표준 (2-bin BEV). NavSim default False라 일탈 |

### 12.4 paper writing 시 disclosure 가이드

- **"NavSim TransFuser baseline + TruckScenes-specific adaptations (12.2) + experimental modifications (12.3)"** 형태로 명시
- §12.2 공통 일탈은 dataset 차이 → reviewer 받아들임
- §12.3 각 v의 추가 일탈은 ablation experiment로 묶어서 보고
- main baseline은 §12.3에서 efficiency·정량 가장 좋은 하나 선택 (현재 후보: v5_truck_only)
- 추가 lever들이 NavSim·paper 표준에서 일탈한 이유 정당화 (§12.2의 7개 + §12.3의 각 v 별 추가)

### 12.5 paper TransFuser (PAMI 2023) 표준과의 추가 차이

NavSim 자체가 paper TransFuser에서 fork되며 변형한 부분 (NavSim 기준이라 우리도 동일하게 일탈):

| 항목 | paper TransFuser | NavSim | 우리 |
|---|---|---|---|
| backbone | RegNetY-3.2GF | ResNet34 | ResNet34 |
| Head | autoregressive GRU + CenterNet | query-based (trajectory + agent + bev_sem) | NavSim 따름 |
| T | 4 | 8 | 8 |
| optimizer | AdamW + weight_decay 0.01 | Adam | Adam (v6_lr_schedule만 AdamW) |
| LR schedule | step decay (×0.1 at ep 30·40) | constant | cosine annealing |
| epochs | 41 | (코드 미명시) | 20 |
| augmentation | ±20° rotation | 없음 | 없음 (real-world dataset이라 sensor re-render 불가) |
| Multi-seed | 3 seed | 3 seed | 1 seed (paper writing 시 추가 학습 필요) |

→ paper TransFuser strict 따르려면 위 차이도 모두 disclose. 단 우리 baseline은 NavSim fork라 NavSim에 가깝게 가는 게 자연스러움.

### 12.6 paper 표준 strict 권장 baseline

paper writing 시 main baseline 후보:
1. **v3_baseline** — NavSim에 가장 가까움. 단 정량 약함 (curvy 1.72)
2. **v5_truck_only** — 정량 best (curvy 1.58, truck 1.00, LiDAR Δ +0.54). 단 BEV range 일탈 + trailer head 제거

→ **v5_truck_only를 main baseline + multi-seed (3 seed) 학습 + §12.2/§12.3 명시 disclose** 가 가장 합리적 strict 패턴.

---

## 13. v7 ground_plane 검증 + 최종 baseline 결정 (2026-05-10)

> **상태**: trailer_v7_ground_plane (use_ground_plane=True) 학습 완료. paper
> TransFuser §3.2 표준 lever 적용. 정량 + LiDAR utilization + paper 충실성
> 종합 평가.
> **결론**: v7이 paper baseline 후보 중 **paper 충실성 + LiDAR utilization
> 가장 좋음**. 정량 truck L2도 best. 단 curvy bin은 v5_truck_only 대비 +0.04m
> noise floor 안 차이.

### 13.1 학습 결과 (val full 2,395)

`configs/v7_ground_plane.py` (= v5_truck_only + use_ground_plane=True). 20 epoch
학습. plateau (ep14-20) 정량:

| metric | v7 ep20 | 비교: v5_truck_only ep20 |
|---|---|---|
| truck full Avg | **0.94** | 1.00 (-0.06 v7 better) |
| straight | 0.95 | 1.00 (-0.05 v7 better) |
| lane | 0.99 | 0.92 (+0.07 v5 better) |
| moderate | 0.88 | 0.95 (-0.07 v7 better) |
| curvy | 1.80 | 1.76 (+0.04 v5 better, ≈ noise) |
| **LiDAR Δ** | **+0.79** | +0.54 (v7 +0.25 better, paper §3.2 표준 효과) |

→ 정량적으로 v5와 거의 동등 (Δ < 0.1m bin별). **차이는 LiDAR utilization
강화에서 명확** — v7의 LiDAR Δ +0.79가 모든 baseline 중 최대.

### 13.2 const-vel 비교 (paper baseline 자격)

| bin | const-vel truck | v7_ep20 truck | Δ | 자격 |
|---|---|---|---|---|
| straight (64%) | 0.34 | 0.95 | +0.61 | const-vel 우세 (trivial mapping) |
| lane (20.4%) | 0.74 | 0.99 | +0.25 | const-vel 우세 |
| **moderate (10%)** | 1.29 | **0.88** | **−0.41** | **v7 우세 ✓** |
| **curvy (5.6%)** | 2.86 | **1.80** | **−1.06** | **v7 우세 ✓** |

→ **moderate·curvy에서 const-vel을 명확히 깸**. paper에서 articulation metric은
곡선 sample 위주라 **paper baseline 자격 충족** ("non-trivial" baseline
= 곡선/회전에서 trivial mapping보다 명확히 좋음).

### 13.3 세 baseline 후보 종합 비교

| 기준 | v3_baseline | v5_truck_only | **v7_ground_plane** |
|---|---|---|---|
| paper TransFuser §3.2 (`use_ground_plane=True`) | ✗ False | ✗ False | **✓ True** |
| NavSim default (`use_ground_plane=False`) | ✓ default | ✓ default | ✗ override |
| trailer head | ✓ 있음 | ✗ 없음 | ✗ 없음 |
| BEV range (forward / backward) | 32 / 32 | 48 / 32 | 48 / 32 |
| truck full L2 (val 2,395) | 1.05 | 1.00 | **0.94** |
| curvy bin truck L2 | **1.72** | 1.76 | 1.80 |
| LiDAR Δ (no_lidar - full) | +0.45 | +0.54 | **+0.79** |
| paper 충실성 (PAMI §3.2) | ✗ | ✗ | **✓** |
| paper 충실성 (NavSim default) | ✓ | △ (range 변경) | △ (ground_plane 변경) |

**Trade-off 정리**:
- **v3_baseline**: NavSim default에 가장 가까움. 정량 약함 (truck 1.05).
- **v5_truck_only**: 정량 좋음 + curvy bin 최고. 단 paper TransFuser와 NavSim
  모두에서 일탈 (BEV range + ground_plane 둘 다 다름).
- **v7_ground_plane**: paper TransFuser §3.2 표준에 정합 + 정량 truck L2 best
  + LiDAR utilization 가장 강력. curvy bin은 v5보다 약간 worse but noise floor 안.

### 13.4 paper baseline 권장: v7_ground_plane

**근거**:
1. **paper TransFuser §3.2의 ground plane 2-bin BEV는 default**. v7만 이걸
   따름 → paper 표준 충실성 가장 강함
2. **LiDAR utilization +0.79 (모든 baseline 중 최대)**. paper에서 LiDAR가
   학습된 모델에 명확히 기여한다고 보고 가능
3. **truck full L2 0.94 (best)** + moderate·curvy bin에서 const-vel 깸 →
   paper baseline 자격 충족
4. curvy bin의 v5와 0.04m 차이는 single-seed noise 영역 (multi-seed로 검증
   필요. paper writing 시점에 추가)

**§12.2 공통 일탈 (TruckScenes 차이) + §12.3 v7 추가 일탈** 명시:
- v7 = NavSim default + (1) §12.2 공통 + (2) `use_ground_plane=True`
  (paper TransFuser §3.2 default로 되돌림) + (3) BEV range -32~48 (TruckScenes
  highway 적응) + (4) `use_trailer_head=False, trailer_weight=0.0` (capacity-matched)

이 disclosure 패턴으로 paper writing.

### 13.5 추가 측정 권고 (paper writing 시)

- **Multi-seed (3 seed) v7_ground_plane** — paper §4.2/§4.7 표준. mean ± std
  로 보고. 1-seed 차이가 noise인지 확인.
- v7 ep20 vs ep14 (curvy best 1.74) ckpt 선택 — paper에 어느 epoch 보고할지
- v7 vs const-vel + 다른 baseline 정량 표 정리

### 13.6 시각적 관찰

`viz/curvy_v7_ground_plane_ep20/` (top 3 curvy scene mp4, idx 12 / 22 / 46).
v5 viz (`viz/curvy_v5_ep20/`)와 직접 비교 가능 (동일 scene).

---

## 14. 측정 재현

```bash
# Const-vel baseline
python tools/checks/check_const_vel_baseline.py

# 곡률 분포
python tools/checks/check_dataset_curvature_dist.py

# Stratified eval — v3 ep12
python tools/checks/check_stratified_eval.py \
    --config v3_baseline \
    --method_name v3_ep12 \
    --checkpoint work_dirs/trailer_v3_status_dropout50_20260428_153540/checkpoints/epoch12.pt

# Stratified eval — v4 ep8 (range 확장 후)
python tools/checks/check_stratified_eval.py \
    --config v4_range \
    --method_name v4_ep8 \
    --checkpoint work_dirs/trailer_v4_forward48_20260430_133921/checkpoints/epoch8.pt

# Curvy scene 시각화 (mp4)
./scripts/visualize_curvy.py \
    --ckpt work_dirs/trailer_v4_forward48_20260430_133921/checkpoints/epoch8.pt \
    --config v4_range --top 3 --out_dir viz/curvy_v4_ep8
```

모든 스크립트 deterministic — 같은 seed/같은 ckpt면 같은 결과 재현됨.
