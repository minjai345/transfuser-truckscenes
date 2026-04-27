# 02. 세 구현 비교 — 공식 CARLA vs NavSim vs 현재 TruckScenes Port

각 행은 **항목 / 공식 CARLA(`/home/minjai/projects/transfuser/team_code_transfuser/`) / NavSim(`navsim/agents/transfuser/`) / 현재 port(`/home/minjai/projects/transfuser-truckscenes/`)** 순. 셋 중 어느 두 칸이 같은지·다른지를 한눈에 비교해 "현재 port가 어디서 왔는지"를 명확히 한다.

> NavSim 코드는 GitHub의 `autonomousvision/navsim` 메인 브랜치 기준 (`navsim/agents/transfuser/transfuser_*.py`). 문서 작성 시점에 raw URL로 직접 확인했다.

---

## 1. 디렉토리·파일 매핑

| 역할 | 공식 CARLA | NavSim | 현재 port |
|---|---|---|---|
| Backbone | `transfuser.py` | `transfuser_backbone.py` | `model/backbone.py` |
| Wrapper / heads | `model.py` (`LidarCenterNet`) | `transfuser_model.py` (`TransfuserModel`) | `model/model.py` (`TransfuserModel`) |
| Loss | `model.py` 내부 + `train.py` | `transfuser_loss.py` (`transfuser_loss`) | `model/loss.py` (`transfuser_loss`) |
| Config | `config.py` (`GlobalConfig`) | `transfuser_config.py` (`TransfuserConfig`) | `model/config.py` (`TransfuserConfig`) |
| Feature/Target build | `data.py` (`CARLA_Data`) | `transfuser_features.py` | `dataset/dataset.py` (단일 Dataset 클래스로 통합) |
| Agent / 학습 wiring | `train.py` (직접) | `transfuser_agent.py` | `train.py` (직접) |
| Enums (StateSE2, BBox2D) | 직접 사용 | `nuplan.*` import + `BoundingBox2DIndex` 자체 정의 | `model/enums.py` (StateSE2Index 추가 자체 정의) |

→ 현재 port는 **NavSim의 7개 파일을 5개로 줄이면서**, nuPlan SDK 의존성을 모두 제거하고 Feature/Target Builder 분리 패턴 대신 단일 `TruckScenesDataset`으로 합쳤다.

---

## 2. 백본 (이미지·LiDAR 인코더)

| 항목 | 공식 CARLA | NavSim | 현재 port |
|---|---|---|---|
| 이미지 백본 | `regnety_032` (RegNetY-3.2GF, ImageNet pretrained) — config.py:50 | `resnet34` (ImNet pretrained) — `transfuser_config.py:image_architecture` | `resnet34` (ImNet pretrained) — config.py:11, backbone.py:25 |
| LiDAR 백본 | `regnety_032` (from scratch) — config.py:52 | `resnet34` (from scratch) | `resnet34` (from scratch) — config.py:12, backbone.py:46-51 |
| LiDAR 입력 채널 | 2 (above/below) 또는 +1 (target_point) | 1 또는 2 (`use_ground_plane`) | 동일 — backbone.py:26-29 |
| 이미지 입력 H×W | 160×704 (3-cam stitch) | 256×1024 (3-cam stitch) | **256×1536** (4-cam stitch) — config.py:31-32 |
| 이미지 anchor grid | 5×22 = 110 tokens — config.py:126-132 | 8×32 — `img_horz_anchors=1024//32` | **8×48** — config.py:36-37 |
| LiDAR anchor grid | 8×8 = 64 tokens | 8×8 | 8×8 — config.py:38-39 |

> 핵심: **백본 종류가 다르다.** 공식은 RegNetY-3.2GF(약 19M params), NavSim·현재 port는 ResNet34(약 21M params로 비슷). 공식이 더 큰 백본을 쓰는 이유는 CARLA의 보조 task(detection/seg/depth)가 많아 표현력을 키울 필요가 있기 때문. NavSim은 task가 단순해 ResNet34로 충분.

---

## 3. Multi-scale GPT Fusion

| 항목 | 공식 CARLA | NavSim | 현재 port |
|---|---|---|---|
| Stage 수 | 4 — config.py:`n_scale=4` | 4 — `n_scale=4` | 4 — config.py:44, backbone.py:61-70 |
| Heads | 4 — config.py:178 | 4 — `n_head=4` | 4 — config.py:43 |
| Layer per stage | 8 (default), 4 (논문 실험) — config.py:177 | **2** — `n_layer=2` | **2** — config.py:42 |
| MLP expansion | 4× | 4× | 4× — config.py:41 (`block_exp=4`) |
| Pos embedding | 학습형 | 학습형 | 학습형 — backbone.py:240-247 |
| Velocity embedding 추가 | 옵션 (config.py:54) | 없음 (status는 별도 토큰으로 들어감) | 없음 (현재 port도 별도 토큰 방식) |

→ NavSim과 현재 port는 layer를 **2**로 가져가서 가볍게 학습 가능하게 만들었다. 공식 default `n_layer=8`을 쓰면 학습 시간이 크게 늘어나지만 표현력도 늘어난다 — 향후 데이터셋 규모가 커지면 검토 가치 있음.

---

## 4. 디코더 / 출력 head

| 항목 | 공식 CARLA | NavSim | 현재 port |
|---|---|---|---|
| Trajectory | GRU autoregressive (T=4, hidden 64) — model.py:611-646 | Transformer decoder + MLP, 단일 query → 8 pose × (x,y,heading) | NavSim 동일 — model.py:141-157 |
| Detection | CenterNet head (heatmap+wh+offset+yaw_cls+yaw_res, 12-bin yaw) — model.py:34-99 | AgentHead (30 query × 5 box + 30 logits, Hungarian) | NavSim 동일 — model.py:117-138, loss.py:48-66 |
| BEV semantic | 3-class CE, weight 1.0 — model.py:581-585, 762 | 7-class CE, weight 10.0 — `bev_semantic_weight=10.0` | head 존재, weight=0 (HD map 없음) — config.py:74, model.py:32-58 |
| Depth | L1, weight 10.0 (multitask 옵션) | 없음 | 없음 |
| Semantic seg | CE, weight 1.0 (multitask 옵션) | 없음 | 없음 |
| Brake / velocity | weight 0.0 (학습 비활성) | 없음 | 없음 |

> NavSim은 closed-loop 제어가 아닌 open-loop trajectory 예측만 평가하므로 GRU autoregressive 대신 **MLP 한 방에 전체 시퀀스**를 뽑는다. 현재 port는 NavSim 방식을 그대로 사용. 따라서 학습 안정성이 좋고 batch size 작아도 잘 수렴한다.

---

## 5. Status / 측정값 입력

| 항목 | 공식 CARLA | NavSim | 현재 port |
|---|---|---|---|
| 입력 차원 | speed scalar (+ velocity embedding 옵션) + target_point(2D, GRU 입력) | **8D** = `driving_command(4) + ego_velocity(2) + ego_acceleration(2)` | **4D** = `[vx, vy, ax, ay]` |
| 어디로 들어가나 | speed → velocity embedding → 모든 GPT 토큰에 add. target_point → GRU 입력 concat 또는 BEV 채널 추가 | `_status_encoding = nn.Linear(4+2+2, tf_d_model)` → keyval token에 add | `_status_encoding = nn.Linear(4, tf_d_model)` — model.py:30 |
| 데이터 출처 | autopilot 측정값 (speed, command_xy 글로벌) | nuPlan AgentInput.ego_statuses | **vx/vy: ego_pose centered difference** (chassis CAN bus는 32% 가짜 0이라 미사용), **ax/ay: chassis IMU** — `_get_status_feature` + `_estimate_ego_velocity`, dataset.py |

→ **가장 큰 의도적 일탈**: TruckScenes에는 routing/driving_command가 없어 4D로 축소. 모델 차원도 `8 → 4`로 함께 수정해야 dimension mismatch가 안 난다(이미 되어 있음, model.py:30).

---

## 6. 카메라 입력

| 항목 | 공식 CARLA | NavSim | 현재 port |
|---|---|---|---|
| 카메라 수 | 3 (front + ±60°), FOV=120° | 3 (`cam_l0`, `cam_f0`, `cam_r0`) | 4 (`CAMERA_LEFT_FRONT`, `RIGHT_FRONT`, `LEFT_BACK`, `RIGHT_BACK`) |
| 원본 해상도 | 960×480 | nuPlan(가변) | 1920×1080 |
| Crop 전략 | 320×160 center crop per cam | `l0[28:-28, 416:-416]`, `f0[28:-28]`, `r0[28:-28, 416:-416]` (4:1 비율 crop) | `_crop_to_aspect(img, 1.5, side=...)`로 1.5:1 비율 → front pair는 directional crop, back pair는 그대로 (dataset.py:127-143, 320-346) |
| 합쳐진 해상도 | 704×160 | 1024×256 | 1536×256 |
| 정규화 | BGR → RGB만, ImageNet norm 없음 | `transforms.ToTensor()`만, ImageNet norm 없음 | `transforms.ToTensor()`만, ImageNet norm 없음 |

→ 셋 모두 ImageNet 정규화 안 함. 일반 ML 모범 사례와 다르지만 reference 구현이 그러하므로 따라감. `04-current-port-review.md` §4.1 (이전 권고는 정정됨).

---

## 7. LiDAR 입력

| 항목 | 공식 CARLA | NavSim | 현재 port |
|---|---|---|---|
| LiDAR 수 | 1 | 1 (nuPlan 통합) | **6** (TOP_FRONT/TOP_LEFT/TOP_RIGHT/LEFT/RIGHT/REAR) — dataset.py:34-40 |
| 좌표계 | ego frame | ego frame | sensor frame → ego frame 변환 후 합침 (dataset.py:158-163) |
| BEV 범위 | ±16m (32m×32m) | ±32m | ±32m — config.py:21-24 |
| 해상도 | 8 px/m → 256² | 4 px/m → 256² | **4 px/m → 256²** (NavSim 동일) — config.py:18 |
| 채널 | above/below 2-bin (+ optional target_point) | above only (default) — `use_ground_plane=False` | above only — config.py:27 |
| 시간 융합 | seq_len=1 (코드는 multi-frame 가능) | seq_len=1 | seq_len=1, **`lidar_time_frames=[1,1,1,1]`이 backbone.py:54에 하드코딩** |

→ 현재 port는 **6개 LiDAR를 합치기 위한 sensor frame 변환이 추가**된 것 외에는 NavSim과 동일한 BEV histogram 표현. 좌표계 변환 로직은 `pyquaternion`으로 sensor→ego 회전 + translation 적용 (dataset.py:159-163).

---

## 8. Trajectory Target 생성

| 항목 | 공식 CARLA | NavSim | 현재 port |
|---|---|---|---|
| Future poses 수 | 4 | 8 (`num_poses`) | 8 — config.py:87 |
| 시간 간격 | 0.5s | 0.5s (`trajectory_sampling.interval`) | 0.5s — config.py:89 |
| 좌표계 | virtual_lidar frame | ego frame | ego frame |
| 변환 방식 | `transform_waypoints()` 함수, ego_matrix 기반 | `Scene.get_future_trajectory()` (nuPlan 추상) | `_quaternion_to_yaw` + 회전행렬 직접 — dataset.py:221-266 |
| Padding | 시퀀스 종료 시 마지막 pose 반복 | 동일 | 시퀀스 종료 시 마지막 pose 반복 — dataset.py:236-241 |

---

## 9. Agent (vehicle) Target 생성

| 항목 | 공식 CARLA | NavSim | 현재 port |
|---|---|---|---|
| 최대 객체 수 | 20 | **30** (`num_bounding_boxes`) | 30 — config.py:68 |
| Box 표현 | (x, y, w, h, yaw, vel, brake) 7-tuple | (x, y, heading, length, width) 5-tuple | (x, y, heading, length, width) 5-tuple — model/enums.py:46-53 |
| Filter | (CARLA에서 type 분기) | `name == "vehicle"` 정확 일치 | `name.startswith("vehicle.")` & `!= "vehicle.ego_trailer"` — dataset.py:45-46, 284 |
| Range filter | 없음(BEV 안 들어가면 자동 무시) | `lidar_min_x..max_x, min_y..max_y` 박스 — `_xy_in_lidar` | 동일 — dataset.py:293-295 |
| Sort/Pad | num_pos별 padding | L2 distance 정렬 후 closest 30 → 0 padding, label boolean | L2 정렬 + closest 30, label float(1.0/0.0) — dataset.py:309-316 |

→ 의도적 일탈: **TruckScenes 카테고리는 계층형 문자열**(`vehicle.car.compact` 등)이라 정확 일치로는 거의 매칭 안 됨. prefix 매칭으로 8가지 vehicle 서브카테고리를 모두 잡고, ego trailer만 제외.

---

## 10. BEV Semantic Target

| 항목 | 공식 CARLA | NavSim | 현재 port |
|---|---|---|---|
| 클래스 수 | 3 (road / lane / other) | 7 (lane, walkway, lane line, static, vehicle, pedestrian, unlabeled=0) | head는 7-class, **target 미생성·loss=0** |
| 출처 | CARLA topdown PNG (`topdown/encoded_*.png`) | nuPlan HD map polygon/linestring + boxes | — (HD map 없음) |
| Loss weight | 1.0 (class_weight `[1,1,3]`) | **10.0** | **0.0** — config.py:74 |

---

## 11. Loss 가중 합

| 항목 | 공식 CARLA | NavSim | 현재 port |
|---|---|---|---|
| 활성 loss | 11개 (waypoint, BEV seg, 5× CenterNet, depth, semantic, velocity=0, brake=0) — config.py:134-136 | 4개 (trajectory, agent_class, agent_box, bev_semantic) | **3개** (trajectory, agent_class, agent_box) |
| 가중치 (대략) | wp=1.0, bev=1.0, det=0.2×5, depth=10.0, sem=1.0 | 10 / 10 / 1 / 10 | 10 / 10 / 1 / **0** — config.py:71-74 |
| 최종 합 | `Σ weight_k * loss_k` | `Σ weight_k * loss_k` | 동일 — loss.py:15-26 |

---

## 12. 학습 schedule

| 항목 | 공식 CARLA | NavSim | 현재 port |
|---|---|---|---|
| Optimizer | AdamW (또는 ZeroRedundancyOptimizer) | Adam (`get_optimizers()`) | Adam — train.py:161 |
| LR | 1e-4 | `self._lr` (caller 지정) | 1e-4 — train.py:339 |
| Schedule | epoch 30, 40에서 ×0.1 (만약 `--schedule 1`) | 없음 | `CosineAnnealingLR(T_max=epochs)` — train.py:162 |
| Augmentation | rotation ±20°, 90% 확률 | 없음 | 없음 |
| Batch size | 12/GPU (DDP) | 미공개 | 4 — train.py:337 |
| Gradient clip | 0.5 (DDP), 1.0 (단일) | callback 의존 | `max_norm=1.0` — train.py:188 |
| Gradient checkpointing | 옵션 | 없음 | 없음 |

---

## 13. 평가 지표

| 항목 | 공식 CARLA | NavSim | 현재 port |
|---|---|---|---|
| 주 지표 | CARLA leaderboard (Driving Score, Route Completion, Infraction) | PDM-score 등 NavSim 자체 메트릭 | **L2 error @ 1/2/3s + Collision rate @ 1/2/3s** — evaluate.py:24, 100-114 |
| 보조 | mAP(detection), Depth L1, IoU(seg) | (callback이 처리) | 없음 |
| 시각화 | submission_agent로 closed-loop run | callback (TransfuserCallback) | `tools/visualize.py`, `tools/predict_video.py` |

---

## 14. 부록 — NavSim이 query-based로 단순화한 이유

CARLA TransFuser는 **closed-loop 자율주행 시뮬레이션**용으로, BEV 위에 detection·segmentation·waypoint를 동시에 잘 뽑아야 routing·제어와 잘 맞물린다. 이를 위해 다양한 보조 head로 backbone에 강한 inductive bias를 부여한다.

NavSim은 **open-loop planning 벤치마크**로, 모델이 future trajectory만 예측하면 PDM-score를 통해 simulation 없이 평가한다. 따라서:
- closed-loop control GRU 대신 MLP 한 번에 시퀀스 생성으로 충분.
- detection도 BEV CenterNet의 dense 예측 대신, **DETR식 query-based 30-token Hungarian matching**으로 단순화.
- depth/semantic seg 같은 sub-task는 학습 비용 대비 이득이 작아 제거.

현재 port는 이 NavSim 라인을 그대로 따르므로, **단순한 trajectory + agent 예측**에 집중되어 있다. CARLA처럼 보조 head로 일반화 성능을 보강하고 싶으면 04 문서 §5에서 언급한 BEV semantic 재활성화나 multi-task head 추가가 필요하다.

---

## 참조 file:line 빠른 색인

- **공식 CARLA**: `/home/minjai/projects/transfuser/team_code_transfuser/{transfuser,model,config,data,train}.py`
- **공식↔논문 매핑**: `/home/minjai/projects/transfuser_paper_code_mapping.md`
- **NavSim raw 파일** (작성 시점):
  - `https://raw.githubusercontent.com/autonomousvision/navsim/main/navsim/agents/transfuser/transfuser_config.py`
  - `https://raw.githubusercontent.com/autonomousvision/navsim/main/navsim/agents/transfuser/transfuser_model.py`
  - `https://raw.githubusercontent.com/autonomousvision/navsim/main/navsim/agents/transfuser/transfuser_features.py`
  - `https://raw.githubusercontent.com/autonomousvision/navsim/main/navsim/agents/transfuser/transfuser_loss.py`
  - `https://raw.githubusercontent.com/autonomousvision/navsim/main/navsim/agents/transfuser/transfuser_agent.py`
- **현재 port**: `/home/minjai/projects/transfuser-truckscenes/{model,dataset}/`, `train.py`, `evaluate.py`
