# TransFuser × MAN TruckScenes

NavSim TransFuser를 [MAN TruckScenes](https://www.man.eu/truckscenes) 데이터셋에 포팅한 연구용 코드베이스. articulation-aware trajectory metric 논문의 baseline.

## 환경 구성

Python 3.11, PyTorch 2.x 환경에서 검증됨.

```bash
# 1. conda env
conda create -n truckscenes python=3.11 -y
conda activate truckscenes

# 2. PyTorch — 본인 시스템 CUDA 버전에 맞춰 설치
#    https://pytorch.org/get-started/locally/ 에서 적절한 명령 확인
pip install torch torchvision

# 3. 본 프로젝트 의존성
pip install -r requirements.txt

# 4. TruckScenes devkit (선택: editable install로 별도 레포에서)
# git clone https://github.com/TUMFTM/truckscenes-devkit.git ../truckscenes-devkit
# pip install -e ../truckscenes-devkit
# 또는 PyPI에서: pip install truckscenes-devkit (requirements.txt에 포함됨)
```

## 데이터셋

TruckScenes v1.1-trainval을 다음 경로에 배치:

```
data/man-truckscenes/
├── samples/
├── sweeps/
└── v1.1-trainval/
    ├── attribute.json
    └── ...
```

다운로드: https://www.man.eu/truckscenes (계정 등록 필요).

## Quick start

```bash
# 학습 (paper baseline)
./scripts/train.py --config v7_ground_plane --run_name v7_run1

# 평가 (val L2 + collision rate + trailer L2)
./scripts/evaluate.py --ckpt work_dirs/v7_run1/checkpoints/epoch20.pt --config v7_ground_plane

# 한 scene 예측 mp4
./scripts/predict_video.py --ckpt <ckpt> --scene_idx 12 --config v7_ground_plane

# val 5장 PNG 시각화
./scripts/visualize.py --ckpt <ckpt> --config v7_ground_plane
```

**Paper baseline = `v7_ground_plane`** (LiDAR 2채널 BEV ground-plane split). 근거는 `docs/06-baseline-validation.md` §13.

## 프로젝트 구조

```
.
├── train.py, evaluate.py   # 실 구현 (argparse + 학습/평가 루프)
├── scripts/                # 얇은 wrapper — DATAROOT/VERSION 하드코딩 + subprocess로 위 호출
├── tools/                  # 독립 분석/시각화 워커 (build_cache, data_stats, visualize, predict_video)
├── model/                  # backbone, model, loss, enums
├── dataset/                # TruckScenesDataset
├── configs/                # v3 ~ v8 실험 config (_base.py에 schema + default)
├── docs/                   # 설계 문서 (아래 참조)
└── data/, work_dirs/, logs/, wandb/, viz/  # 산출물 (gitignored)
```

- **`scripts/` vs `tools/`**: scripts는 default 채워 한 명령으로 도는 wrapper. tools는 풀 argparse가 있는 독립 워커. `scripts/predict_video.py`는 내부적으로 `tools/predict_video.py`를 subprocess로 호출.

## 다음 읽기

| 목적 | 문서 |
|---|---|
| 모델 아키텍처 (TransFuser PAMI 2023) | `docs/01-architecture.md` |
| NavSim/CARLA reference와 비교 | `docs/02-reference-implementations.md` |
| TruckScenes 적응 (센서/좌표/target) | `docs/03-truckscenes-adaptation.md` |
| 현재 코드 정합성 검토 | `docs/04-current-port-review.md` |
| 학습/평가 실행 체크리스트 | `docs/05-implementation-checklist.md` |
| baseline 검증 (v3~v8 결과 + 권장) | `docs/06-baseline-validation.md` |
| config 버전 이력 | `configs/README.md` |

이전 작업 일지(2026-04 스냅샷)는 `docs/archive/`.
