# 05. 학습/평가 실행 체크리스트

> 코드를 읽지 않고도 학습·평가·시각화를 돌릴 수 있도록 정리한 실용 가이드. "정상 학습"의 형태와 자주 마주치는 함정을 함께 적었다.

---

## 1. 환경

- Python 3.7+ (truckscenes-devkit은 3.8+ 권장)
- CUDA 12.x 호환 GPU 1장 이상 (검증 환경: RTX A6000)
- 주요 의존성: `torch`, `torchvision`, `timm`, `opencv-python`, `pyquaternion`, `truckscenes-devkit`, `wandb`

```bash
cd /home/minjai/projects/transfuser-truckscenes
pip install -r requirements.txt
pip install -e ../truckscenes-devkit  # devkit이 별도 패키지로 설치 필요
```

학습 시 wandb를 쓰지 않으려면 `WANDB_MODE=offline` 환경변수.

---

## 2. 데이터 레이아웃

```
transfuser-truckscenes/
└── data/
    └── man-truckscenes/
        ├── samples/      # 키프레임 데이터 (이미지, LiDAR 등)
        ├── sweeps/       # 비키프레임 데이터
        ├── v1.1-test/    # 테스트 split metadata JSON
        └── v1.1-trainval # train+val split metadata JSON
            └── (mini 사용 시 v1.0-mini로 대체)
```

- `TruckScenes(version="v1.0-mini", dataroot="data/man-truckscenes")` 식으로 dev kit이 로드.
- mini로 동작 확인 후 v1.1로 확장 권장.

---

## 3. 정상성 검증 — Sanity Check 단계

### 3.1 데이터 통계 (`tools/data_stats.py`)
- scene·sample 수, vehicle 카테고리 분포, LiDAR 포인트 수 등 출력.
- 카테고리 분포 출력에서 `vehicle.*` 외에 다른 prefix가 학습 target으로 들어가지 않는지 확인.

### 3.2 1 batch overfit (`tools/checks/overfit_test.py`)
- 단일 batch(작게는 1~4 sample)를 반복 학습시켜 loss가 빠르게 0 근처로 떨어지는지 확인.
- **기대 동작**: trajectory loss + agent class/box loss 합이 50 epoch 이내로 한 자리 수까지 떨어진다. 최근 커밋 `4da6d12`로 trajectory/class/box loss를 분해해 출력하므로 어느 component가 오버피팅 안 되는지 바로 식별 가능.
- **함정**:
  - status_feature가 항상 0이면 `_status_encoding` 출력이 동일해 정보 손실. CAN bus가 비어있는지 확인.
  - trajectory target이 모두 (0,0,0)으로 패딩만 들어오면 모델이 trivial 해를 찾는다. `dataset.py:236-241`의 padding 분기에 들어가는 sample이 너무 많지 않은지 확인.
  - agent_labels이 모두 0이면 Hungarian loss가 0이 되는 trivial 상태. v1.0-mini에서도 vehicle은 충분히 많아야 정상.

### 3.3 시각화 점검 (`tools/visualize.py`, `tools/predict_video.py`)
- BEV histogram이 ego 중심에서 좌우/전후로 잘 펼쳐져 있는지 (회전 행렬 적용 누락 시 한 쪽으로 쏠림).
- 4-cam stitching이 좌→우(LF, RF, LB, RB) 순으로 자연스럽게 이어지는지.
- GT trajectory를 BEV에 그려봤을 때 ego가 (0,0)에서 출발해 매끄럽게 진행하는지. yaw 변환 부호가 잘못되면 trajectory가 좌우 반전됨.

`1145f1f` 커밋으로 4-cam GT/예측 box 오버레이가 추가됐고, `65967b6`로 BEV orientation·라벨 표시가 정렬됐다 — 시각화 결과를 그대로 신뢰해도 됨.

---

## 4. 본 학습 실행

### 4.1 권장: scripts/ 래퍼 사용
Python wrapper(`scripts/`)가 default(dataroot, batch, num_workers, --wandb)를 모두 채워준다. shebang에 conda env 절대경로가 박혀 있어 `./scripts/X.py`로 직접 실행 가능.

```bash
# 새 학습 (기본 trailer_v2, 20 epochs, batch 8, num_workers 8, wandb on)
./scripts/train.py
./scripts/train.py --run_name my_exp --epochs 30

# Resume — ckpt 경로에서 work_dir/run_name 자동 추정
./scripts/train.py --resume work_dirs/<run>/checkpoints/epoch5.pt
./scripts/train.py --resume <ckpt> --epochs 30   # total epochs 변경

# 시각화 / 영상 / 평가
./scripts/visualize.py --ckpt <ckpt> --num 10
./scripts/predict_video.py --ckpt <ckpt> --scene_idx 12
./scripts/evaluate.py --ckpt <ckpt>
```

### 4.2 직접 실행 (저수준)
필요하면 `train.py` 직접 호출도 동일하게 가능:
```bash
python train.py \
  --dataroot data/man-truckscenes \
  --version v1.1-trainval \
  --batch_size 8 \
  --num_workers 8 \
  --lr 1e-4 \
  --epochs 20 \
  --wandb --wandb_run_name my_exp
```

### 4.3 Resume 메커니즘
ckpt에 다음이 저장됨 (train.py):
- `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`
- `epoch`, `global_step` (wandb step 연속성)
- `wandb_run_id` (같은 run으로 이어가기 — `wandb.init(id=..., resume="must")`)

`--resume <ckpt>` 시:
- ckpt의 work_dir 자동 사용 (parent.parent)
- model/optimizer/scheduler/step/epoch 모두 복원
- 옛 ckpt(scheduler/step 안 저장)는 epoch만큼 step()로 LR 추정 (정확도 미세 차이)

### 4.2 work_dir 구조
`835fcd7` 커밋 이후 모든 산출물이 정리됨:
```
work_dirs/<run_name>_<timestamp>/
├── checkpoints/
│   └── epoch_XX.pt
├── logs/
│   └── train.log         # stdout/stderr 기록
└── wandb/                # wandb 로컬 캐시
```

### 4.3 권장 하이퍼파라미터 (mini 기준)

| 항목 | 값 | 비고 |
|---|---|---|
| batch_size | 4 | A6000 24GB에서 안정 |
| lr | 1e-4 | NavSim과 동일 |
| epochs | 10~30 | 5 epoch에 loss 68→30 (REPORT §9) → 30 epoch까지 더 떨어짐 기대 |
| optimizer | Adam | train.py:161 |
| scheduler | CosineAnnealingLR (T_max=epochs) | train.py:162 |
| grad clip | 1.0 | train.py:188 |
| num_workers | 4~8 | mini는 4 충분, trainval은 8+ |

> v1.1 trainval 확장 시: `04-current-port-review.md` §4.5의 캐시 권고를 먼저 반영하길 권장. 안 그러면 디스크 I/O가 학습 시간을 지배함.

---

## 5. 평가 실행

### 5.1 명령
`evaluate.py`는 학습 epoch 종료 시 자동 호출되지만, 단독 실행도 가능:
```bash
python evaluate.py \
  --dataroot data/man-truckscenes \
  --version v1.0-mini \
  --checkpoint work_dirs/<run>/checkpoints/epoch_XX.pt
```

### 5.2 출력 메트릭

| 키 | 의미 | "정상" 범위 (mini) |
|---|---|---|
| `l2/1s` | 1초 후 ego 위치 예측 오차 (m) | 0.5 ~ 1.5m 정도면 합리적 |
| `l2/2s` | 2초 후 | 1 ~ 3m |
| `l2/3s` | 3초 후 | 2 ~ 5m |
| `l2/avg` | 평균 | 위 셋의 평균 |
| `col/1s`, `2s`, `3s` | 충돌 발생 비율 (0~1) | 0에 가까울수록 좋음 |

> Ego 차원: `ego_length=6.9m, ego_width=2.5m` (evaluate.py:123). 실제 트럭이라 충돌 박스가 큼. mini의 짧은 학습으로는 col이 다소 높게 나올 수 있음 — 학습 스텝/데이터를 늘리면 개선.

### 5.3 함정
- evaluate.py:136 부근에 `interval=0.5s` 가정이 하드코딩되어 있음. config 변경 시 평가 결과 인덱스가 어긋날 수 있음 (`04-current-port-review.md` §4.4).
- collision은 Shapely Polygon 교차로 판정. ego·agent box의 yaw가 같이 회전해야 정확. 시각화로 잘못된 사례 1~2건은 확인할 것.

---

## 6. "정상" 학습 곡선의 형태

REPORT.md §9의 mini 학습 결과를 기준점으로:

| Epoch | Avg loss | 의미 |
|---|---|---|
| 1 | 68.01 | 거의 random 초기, agent loss가 큰 비중 |
| 2 | 57.35 | trajectory가 큰 폭으로 개선되기 시작 |
| 3 | 46.38 | agent_class loss 떨어지기 시작 |
| 4 | 40.22 | agent_box loss도 함께 떨어짐 |
| 5 | 29.93 | 합산이 안정적으로 한 자릿수대 |

**경고 신호**:
- 첫 3 epoch 동안 loss가 거의 그대로면 lr이 너무 작거나 trajectory target 분포 문제.
- trajectory loss는 떨어지는데 agent loss가 안 떨어지면 vehicle filter가 잘못되어 target이 비어 있을 가능성 (REPORT.md §10의 vehicle filter 버그 사례 참고).
- loss가 NaN이면 quaternion 정규화 문제일 가능성 (`pyquaternion`은 자동 정규화하지만 외부 입력이 비정상이면 발산 가능).

---

## 7. 디버깅 팁

| 증상 | 원인 후보 | 빠른 확인 방법 |
|---|---|---|
| BEV histogram이 한 쪽으로 쏠림 | sensor→ego 변환 누락 또는 quaternion 순서 오류 | dataset.py:159-163 직접 점검, `tools/visualize.py`로 BEV 출력 |
| Trajectory가 항상 직선 | trajectory target이 padding으로 가득 | `_collect_valid_samples`에 들어간 sample 수 vs scene 길이 출력 |
| Agent loss가 0이거나 NaN | 모든 agent_labels이 0 또는 모든 box가 range 밖 | dataset.py:282-303의 필터 결과를 batch 1개에 대해 출력 |
| Loss 합이 음수가 되거나 발산 | quaternion이 정규화 안 됨, NaN trajectory | `torch.isnan(targets["trajectory"]).any()` 점검 |
| 학습이 빠르게 trivial 수렴 | status_feature가 항상 0, agent target이 비어 있음 | overfit_test에서 component별 loss 출력 (4da6d12 커밋 기능) |
| 카메라가 잘못 stitching | crop side 인자가 swap | dataset.py:26-30의 (channel, crop_side) 매핑 확인 |

---

## 8. 다음 작업 제안 (이 문서 외부)

- **`04-current-port-review.md` §4.1, §4.2 권고를 적용**한 비교 실험 (정규화 + augmentation의 효과 측정)
- v1.1 trainval로 스케일 업 시 데이터 캐시 도입
- BEV semantic을 위한 거친 lane mask 확보 가능성 검토 (예: ego pose 기반 OSM 추출)

---

## 9. 빠른 명령 cheat sheet

```bash
# 환경
cd /home/minjai/projects/transfuser-truckscenes
export WANDB_MODE=offline   # wandb 로그인 없이 시험할 때

# 데이터 통계
python tools/data_stats.py --dataroot data/man-truckscenes --version v1.0-mini

# 1 batch overfit
python tools/checks/overfit_test.py --dataroot data/man-truckscenes --version v1.0-mini

# 학습
python train.py --dataroot data/man-truckscenes --version v1.0-mini --batch_size 4 --epochs 10

# 평가
python evaluate.py --dataroot data/man-truckscenes --version v1.0-mini \
    --checkpoint work_dirs/<run>/checkpoints/epoch_XX.pt

# 시각화 (예측 비디오)
python tools/predict_video.py --dataroot data/man-truckscenes --version v1.0-mini \
    --checkpoint work_dirs/<run>/checkpoints/epoch_XX.pt --scene <scene_token>
```

(각 도구의 정확한 인자는 `--help`로 확인.)
