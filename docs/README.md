# TransFuser × MAN TruckScenes 구현 문서

이 폴더는 `transfuser-truckscenes/` 프로젝트가 **TransFuser 논문 / 공식 CARLA 레포 / NavSim 레포의 TransFuser 모듈**을 어떤 식으로 차용·변형해서 MAN TruckScenes에 적용했는지를 설명한다. 그리고 "지금의 구현이 정합적인지" 점검한 결과와, "더 제대로 만들려면 무엇을 추가해야 하는지" 권고를 함께 정리한다.

> 톤·구성은 기존 `REPORT.md`(루트), `transfuser_paper_code_mapping.md`(상위 워크스페이스)와 같은 한국어 본문 + 영어 코드/식별자 스타일을 유지한다.

---

## 누가 어떤 순서로 읽으면 되나

| 독자 | 읽을 순서 |
|---|---|
| **TransFuser를 처음 보는 사람** | `01-architecture.md` → `02-reference-implementations.md` → `03-truckscenes-adaptation.md` |
| **현재 코드를 수정하려는 사람** | `02-reference-implementations.md` → `04-current-port-review.md` → `05-implementation-checklist.md` |
| **학습/평가만 돌릴 사람** | `05-implementation-checklist.md` → 문제 생기면 `04-current-port-review.md` |
| **연구 보고서 쓸 사람** | 루트의 `REPORT.md` + `01-architecture.md` + `02-reference-implementations.md` |

---

## 문서 구성

| 파일 | 한 줄 설명 |
|---|---|
| [`01-architecture.md`](01-architecture.md) | TransFuser 논문(PAMI 2023)의 모델 아키텍처를 정리. Multi-scale GPT fusion, head 구성, loss 정의 |
| [`02-reference-implementations.md`](02-reference-implementations.md) | **공식 CARLA 레포 / NavSim 레포 / 현재 TruckScenes port** 세 구현을 항목별 표로 비교 |
| [`03-truckscenes-adaptation.md`](03-truckscenes-adaptation.md) | TruckScenes에 어떻게 적용했는지 단계별 가이드 (센서 매핑·데이터 파이프라인·target 생성·좌표계) |
| [`04-current-port-review.md`](04-current-port-review.md) | 현재 코드 검토 결과: NavSim과 정합한 부분 / 의도적 일탈 / 발견된 갭 / 개선 권고 |
| [`05-implementation-checklist.md`](05-implementation-checklist.md) | 학습·평가·시각화 실행 체크리스트와 정상 학습 곡선 형태 |

---

## 외부 참조

- **논문**: "TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving" (PAMI 2023)
  - 워크스페이스 PDF: `/home/minjai/projects/transfuser/TransFuser: Imitation with Transformer-Based.pdf`
- **공식 CARLA 레포**: `/home/minjai/projects/transfuser/`
  - 핵심 디렉토리: `team_code_transfuser/{transfuser,model,config,data,train}.py`
  - 한 줄 매핑 문서: `/home/minjai/projects/transfuser_paper_code_mapping.md`
- **NavSim 레포**: <https://github.com/autonomousvision/navsim>
  - 핵심 경로: `navsim/agents/transfuser/transfuser_*.py`
- **현재 port**: `/home/minjai/projects/transfuser-truckscenes/`
  - 모델: `model/{backbone,model,loss,config,enums}.py`
  - 데이터: `dataset/dataset.py`
  - 실행: `train.py`, `evaluate.py`
  - 도구: `tools/{overfit_test,visualize,predict_video,data_stats}.py`
  - 일회성 검증/디버깅 스크립트: `tools/checks/` (gitignored)
  - 실행 wrapper (default 인자 채움): `scripts/{train,visualize,predict_video,evaluate}.py`

---

## 한눈에 보는 요약

| 차원 | 결론 |
|---|---|
| 백본 (ResNet34 × ResNet34, 4-stage GPT) | NavSim과 동일하게 유지 — 변형 없음 ✅ |
| 출력 head (Trajectory + AgentHead) | NavSim과 동일. BEV semantic head는 살아있으나 **loss weight = 0** (HD map 부재) |
| Status feature | NavSim 8D(driving_command + vel + acc) → 현재 **4D(vel + acc)**. vx/vy는 **ego_pose 미분** (chassis CAN bus는 32% 가짜 0이라 사용 X). ax/ay는 chassis IMU 그대로. `_status_encoding` Linear 차원도 함께 수정 |
| 카메라 stitching | NavSim 3-cam 1024×256 → 현재 **4-cam 1536×256** (CAM_LEFT_FRONT/RIGHT_FRONT/LEFT_BACK/RIGHT_BACK) |
| LiDAR | NavSim 1-cloud → 현재 **6-LiDAR ego frame 변환 후 합치고 BEV histogram** |
| Trajectory target | NavSim `Scene.get_future_trajectory()` → 현재 **연속 sample의 ego_pose chain + quaternion 변환** |
| Agent target | NavSim `name == "vehicle"` → 현재 **`vehicle.*` prefix 매칭** (단, `vehicle.ego_trailer` 제외) |
| BEV semantic | HD map 부재 → 비활성화 (head·target 모두 미사용) |
| Augmentation | NavSim도 안 쓰지만 공식 CARLA는 사용. 현재 port도 미적용 — 개선 후보 |
| ImageNet 정규화 | **NavSim/CARLA 모두 미적용** → 일관성 위해 우리도 미적용 (ToTensor만). reference 따라감 |
| **Trailer head** | NavSim에 없음. 우리가 추가 — articulated truck 위해 **truck/trailer trajectory 따로 출력** (model.py: `_trajectory_head` + `_trailer_trajectory_head`, loss.py: `truck_l1` + `trailer_l1` mask 처리) |
| **Resume** | ckpt에 `scheduler_state_dict + global_step + wandb_run_id` 저장 → `--resume <ckpt>`로 같은 work_dir + 같은 wandb run 이어가기 |

상세는 `02-reference-implementations.md`와 `04-current-port-review.md` 참조.
