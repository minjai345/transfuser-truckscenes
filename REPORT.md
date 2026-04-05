# TransFuser Adaptation for MAN TruckScenes

## Overview

NAVSIM 레포의 TransFuser 모듈을 MAN TruckScenes 데이터셋에서 학습 가능하도록 별도 프로젝트로 추출 및 변형하였다. 핵심 모델 아키텍처(backbone, transformer decoder, prediction heads)는 유지하면서, 데이터 로딩 파이프라인과 nuPlan 의존성을 전면 교체하였다.

---

## 1. nuPlan 의존성 제거

원본 TransFuser는 nuPlan SDK에 강하게 결합되어 있다. 아래 의존성을 모두 자체 구현으로 대체하였다.

| 원본 (nuPlan) | 대체 | 비고 |
|---|---|---|
| `nuplan.common.actor_state.tracked_objects_types.TrackedObjectType` | 제거 | TruckScenes 카테고리명을 직접 문자열 매칭 |
| `nuplan.common.maps.abstract_map.SemanticMapLayer` | 제거 | HD map 미사용 |
| `nuplan.common.actor_state.oriented_box.OrientedBox` | 제거 | bbox를 (x, y, heading, length, width) 배열로 직접 처리 |
| `nuplan.common.actor_state.state_representation.StateSE2` | `transfuser/enums.py::StateSE2Index` | IntEnum으로 인덱싱만 담당 |
| `nuplan.planning.simulation.trajectory.trajectory_sampling.TrajectorySampling` | `TransfuserConfig.num_poses` | config 필드로 단순화 |
| `nuplan.database.maps_db.gpkg_mapsdb.MAP_LOCATIONS` | 제거 | map 미사용 |
| `nuplan.common.maps.nuplan_map.map_factory.get_maps_api` | 제거 | map 미사용 |
| `LidarPointCloud` (nuPlan) | `truckscenes.utils.data_classes.LidarPointCloud` | TruckScenes devkit의 PCD 로더 사용 |

**변경 파일**: `transfuser/enums.py` (신규), `transfuser/transfuser_config.py`, `transfuser/transfuser_model.py`, `transfuser/transfuser_loss.py`

---

## 2. 센서 구성 변경

### 2.1 카메라

| 항목 | NAVSIM (원본) | TruckScenes (변경) |
|---|---|---|
| 카메라 수 | 3개 (cam_l0, cam_f0, cam_r0) | 4개 (LEFT_FRONT, RIGHT_FRONT, LEFT_BACK, RIGHT_BACK) |
| 전방 중앙 카메라 | cam_f0 (존재) | 없음 (좌/우 전방으로 대체) |
| 후방 카메라 | 미사용 | CAMERA_RIGHT_BACK 사용 |
| 원본 해상도 | 다양 (nuPlan) | 1920x1080 (TruckScenes) |
| Stitching 방식 | L-F-R 3개 crop 후 가로 연결, 1024x256 | 4개 각각 1.5:1 center crop 후 가로 연결, 1536x256 |

**변경 사항**:
- `camera_width`: 1024 → 1536 (4카메라 stitching에 맞게 확대)
- `img_horz_anchors`: 1024//32 → 1536//32 (backbone anchor grid 조정)
- Center crop 함수를 범용적으로 재작성하여 임의 해상도에 대응

### 2.2 LiDAR

| 항목 | NAVSIM (원본) | TruckScenes (변경) |
|---|---|---|
| LiDAR 수 | 1개 | 6개 (TOP_FRONT, TOP_LEFT, TOP_RIGHT, LEFT, RIGHT, REAR) |
| 파일 포맷 | nuPlan PCD (바이너리) | TruckScenes PCD (binary_compressed) |
| 로딩 방식 | `nuplan.LidarPointCloud.from_buffer()` | `truckscenes.LidarPointCloud.from_file()` |
| 포인트 형식 | (6, N): x,y,z,intensity,ring,lidar_id | (4, N): x,y,z,intensity |
| 좌표계 | ego frame | sensor frame → ego frame 변환 필요 |

**변경 사항**:
- 6개 LiDAR를 각각 로드 후 calibrated_sensor 정보로 ego frame 변환
- 변환된 포인트 클라우드를 병합하여 하나의 BEV histogram으로 생성
- histogram 생성 로직 자체(splat_points)는 원본과 동일하게 유지

---

## 3. 데이터 파이프라인 변경

### 3.1 원본 (NAVSIM)

```
Pickle logs → SceneLoader → Scene.from_scene_dict_list()
  → AgentInput (sensor data) + Scene (ground truth)
  → TransfuserFeatureBuilder.compute_features()
  → TransfuserTargetBuilder.compute_targets()
  → gzip 캐시 저장
```

- 자체 pickle 포맷으로 사전 변환된 데이터 사용
- Feature/Target Builder 패턴으로 분리

### 3.2 변경 (TruckScenes)

```
TruckScenes DB → TruckScenes devkit API
  → TruckScenesDataset.__getitem__()
    → _get_camera_feature()   : 4 카메라 로드 + stitch
    → _get_lidar_feature()    : 6 LiDAR 로드 + merge + histogram
    → _get_status_feature()   : ego pose로부터 속도 계산
    → _get_trajectory_target(): 미래 ego pose로 trajectory 생성
    → _get_agent_targets()    : 3D bbox를 ego frame 2D로 변환
```

- pickle 전처리 없이 devkit API로 직접 데이터 접근
- Feature/Target Builder 분리 패턴 대신 단일 Dataset 클래스로 통합
- 캐시 미사용 (mini dataset 규모에서는 불필요)

---

## 4. Ego Status 처리

| 항목 | NAVSIM (원본) | TruckScenes (변경) |
|---|---|---|
| driving_command | pickle에 포함 (정수 → one-hot 4D) | 미제공 → dummy [1,0,0,0] 사용 |
| velocity (vx, vy) | pickle의 `ego_dynamic_state`에서 직접 로드 | 연속 프레임 ego pose의 수치 미분으로 계산 |
| acceleration (ax, ay) | pickle의 `ego_dynamic_state`에서 직접 로드 | 현재 dummy [0,0] (추후 2차 미분 추가 가능) |
| 좌표계 | global frame에서 제공 | global ego pose에서 계산 |

---

## 5. Target 생성 변경

### 5.1 Trajectory Target

| 항목 | NAVSIM (원본) | TruckScenes (변경) |
|---|---|---|
| 소스 | Scene 객체의 `get_future_trajectory()` | 연속 sample의 ego_pose를 순회 |
| 좌표 변환 | 내부적으로 global → local 변환 | quaternion 기반 global → ego-centric 변환 직접 구현 |
| 프레임 수 | `TrajectorySampling.num_poses` (default: 8) | `TransfuserConfig.num_poses` (default: 8) |

### 5.2 Agent Detection Target

| 항목 | NAVSIM (원본) | TruckScenes (변경) |
|---|---|---|
| 소스 | `Annotations.boxes` + `Annotations.names` | `ts.get_boxes()` → Box3D 객체 리스트 |
| 필터링 | `name == "vehicle"` | `name.split(".")[0] in VEHICLE_CATEGORIES` (8개 카테고리) |
| 좌표 변환 | 이미 ego frame | global → ego frame 변환 (quaternion rotation) |
| Box 형식 | (x,y,z,l,w,h,yaw) → (x,y,heading,l,w) | Box3D.center/wlh/orientation → (x,y,heading,l,w) |

### 5.3 BEV Semantic Map Target

| 항목 | NAVSIM (원본) | TruckScenes (변경) |
|---|---|---|
| road, walkway, centerline | nuPlan HD map API로 생성 | **제거** (HD map 미제공) |
| vehicle, pedestrian boxes | annotation에서 생성 | 모델에 head는 남아있으나 loss weight = 0 |
| loss weight | 10.0 | **0.0** (비활성화) |

---

## 6. Config 변경 요약

| 파라미터 | 원본 값 | 변경 값 | 사유 |
|---|---|---|---|
| `camera_width` | 1024 | 1536 | 4카메라 stitching (6:1) |
| `img_horz_anchors` | 1024//32 = 32 | 1536//32 = 48 | camera_width에 맞춤 |
| `use_bev_semantic` | True | False | HD map 없음 |
| `bev_semantic_weight` | 10.0 | 0.0 | BEV semantic loss 비활성화 |
| `num_poses` | (TrajectorySampling에서) | 8 (config 직접 보유) | nuPlan 의존성 제거 |

---

## 7. Loss 함수 변경

```python
# 원본: 4개 loss의 가중합
loss = trajectory_loss + agent_class_loss + agent_box_loss + bev_semantic_loss

# 변경: BEV semantic loss를 조건부로 분리 (weight=0이면 미계산)
loss = trajectory_loss + agent_class_loss + agent_box_loss
if config.bev_semantic_weight > 0 and "bev_semantic_map" in targets:
    loss += bev_semantic_loss
```

BEV semantic head는 모델에 그대로 남아있어 향후 map 데이터가 확보되면 weight만 조정하여 바로 활성화 가능.

---

## 8. 모델 아키텍처 변경 없음

아래 모듈은 원본과 동일하게 유지:
- **TransfuserBackbone**: ResNet34 이미지/LiDAR 인코더 + 4-level GPT fusion
- **Transformer Decoder**: 3-layer, 8-head, d_model=256
- **TrajectoryHead**: 8-pose (x, y, heading) 예측
- **AgentHead**: 30개 bbox + confidence 예측 (Hungarian matching loss)
- **BEV Semantic Head**: 7-class segmentation (weight=0으로 비활성화만)

import 경로만 `navsim.agents.transfuser.*` → `transfuser.*`로 변경.

---

## 9. 학습 결과 (v1.1-mini, 320 samples)

| Epoch | Avg Loss | Time |
|-------|----------|------|
| 1 | 68.01 | 16.6s |
| 2 | 57.35 | 14.4s |
| 3 | 46.38 | 14.1s |
| 4 | 40.22 | 13.9s |
| 5 | 29.93 | 14.4s |

- batch_size=4, lr=1e-4, Adam optimizer
- GPU: NVIDIA RTX A6000, CUDA 12.8
- Loss가 5 epoch 동안 68 → 30으로 지속 감소 확인

---

## 10. 프로젝트 구조

```
transfuser-truckscenes/
├── transfuser/
│   ├── enums.py                 # StateSE2Index, BoundingBox2DIndex (nuPlan 대체)
│   ├── transfuser_config.py     # TruckScenes 맞춤 설정
│   ├── transfuser_model.py      # 모델 (아키텍처 동일, import만 변경)
│   ├── transfuser_backbone.py   # 백본 (동일, import만 변경)
│   └── transfuser_loss.py       # Loss (BEV semantic 조건부 비활성화)
├── data/
│   └── dataset.py               # TruckScenes devkit 기반 Dataset (신규)
├── train.py                     # 학습/sanity check 스크립트 (신규)
└── requirements.txt
```
