# 03. TruckScenes 적용 가이드

> "NavSim 기반의 TransFuser 코드를 MAN TruckScenes에서 학습 가능하도록 만들 때 어떤 부분을 바꿨고, 왜 그렇게 바꿔야 하는가"를 단계별로 설명한다. 이미 구현은 완료되어 있으므로(`/home/minjai/projects/transfuser-truckscenes/`) 이 문서는 (1) 의사결정의 근거, (2) 향후 다른 트럭/로봇 데이터셋으로 옮길 때 참고할 체크리스트, 두 목적을 갖는다.

---

## 0. 출발점

NavSim의 TransFuser는 **nuPlan SDK 기반 + pickle 사전 변환 데이터**를 전제로 한다. TruckScenes는 nuScenes-like devkit을 제공하지만 nuPlan은 아니다. 따라서 다음 4가지가 모두 필요하다:

1. nuPlan 의존성 제거 (모델·loss·config·enums)
2. 센서 구성 차이 흡수 (4 카메라, 6 LiDAR)
3. nuPlan pickle → TruckScenes devkit API로 데이터 파이프라인 교체
4. driving_command가 부재한 상황에서 status feature 재정의

---

## 1. nuPlan 의존성 제거

| 원본 (nuPlan) | 처리 | 비고 |
|---|---|---|
| `TrackedObjectType` | 제거 | 카테고리명 문자열 매칭(`startswith("vehicle.")`)으로 대체 |
| `SemanticMapLayer` | 제거 | HD map 미사용 |
| `OrientedBox` | 제거 | Box를 5-tuple `(x, y, heading, length, width)` numpy array로 직접 처리 |
| `StateSE2` | `model/enums.py::StateSE2Index` | IntEnum으로 인덱싱만 담당, 객체 대신 numpy array 사용 |
| `TrajectorySampling` | `TransfuserConfig.num_poses` 등 단순 필드 | 기본값 `num_poses=8`, `interval=0.5s`, `time=4.0s` |
| `MAP_LOCATIONS`, `get_maps_api` | 제거 | map 미사용 |
| `LidarPointCloud` (nuPlan) | `truckscenes.utils.data_classes.LidarPointCloud` | TruckScenes devkit의 PCD 로더 |

`model/enums.py`(97 라인)는 `StateSE2Index`(x,y,heading)와 `BoundingBox2DIndex`(x,y,heading,length,width) 두 IntEnum을 정의하고, `size()`, `POINT`, `STATE_SE2` 같은 NavSim과 동일한 헬퍼 프로퍼티를 노출한다. 모델·dataset·loss 모두 이 enum의 인덱스만 사용하기 때문에, **NavSim 원본 코드에서 변경되는 줄이 거의 없다**.

---

## 2. 센서 구성 매핑

### 2.1 카메라 (4-cam, 1536×256 stitching)

NavSim이 사용한 nuPlan 카메라는 정면 중심 3-cam(`cam_l0/f0/r0`)이지만 TruckScenes는 정면 중앙 카메라가 없고 **좌·우 전방** + **좌·우 후방** 4개로 구성된다. 따라서:

```python
# dataset.py:26-30
CAMERA_CHANNELS = [
    ("CAMERA_LEFT_FRONT",  "left"),   # 우측을 잘라 외측 FOV 유지
    ("CAMERA_RIGHT_FRONT", "right"),  # 좌측을 잘라 외측 FOV 유지
    ("CAMERA_LEFT_BACK",   None),     # crop 없음
    ("CAMERA_RIGHT_BACK",  None),     # crop 없음
]
```

- 각 카메라 1920×1080 → `_crop_to_aspect(img, 1.5, side)`로 1.5:1 비율 crop. front pair는 directional crop(외측 FOV 보존), back pair는 그대로. 이렇게 4장을 가로로 concat → resize 1536×256.
- camera_width=1536 (config.py:31)이 되면서 backbone의 `img_horz_anchors`도 1024//32 → **1536//32 = 48**로 함께 변경 (config.py:37).

> **왜 directional crop인가**: 후방 카메라에는 외측 FOV가 별로 의미 없고(트레일러 가림), 전방은 좌측 카메라의 좌측·우측 카메라의 우측이 가장 중요한 외곽 시야이기 때문. center crop을 하면 두 전방 카메라의 시야가 겹쳐 손실이 크다.

### 2.2 LiDAR (6 sensor, ego frame 합치기)

TruckScenes는 6개 LiDAR(상부 3개 + 측면 2개 + 후면 1개)를 제공한다. NavSim/공식 CARLA는 ego frame에서 단일 cloud를 받는다고 가정하므로, 다음을 한다:

```python
# dataset.py:145-171 요약
for channel in LIDAR_CHANNELS:                        # 6 sensors
    pc = LidarPointCloud.from_file(path)              # (4, N): xyz + intensity
    cs = ts.get("calibrated_sensor", token)
    R = Quaternion(cs["rotation"]).rotation_matrix
    t = np.array(cs["translation"])
    points_xyz = pc.points[:3].T @ R.T + t            # sensor → ego
    all_points.append(points_xyz)
merged = np.concatenate(all_points, axis=0)
return self._compute_lidar_histogram(merged)          # NavSim과 동일한 splat_points 로직
```

- **포인트 형식**: TruckScenes PCD는 `(4, N)` (x, y, z, intensity), NavSim의 nuPlan은 `(6, N)` (x, y, z, intensity, ring, lidar_id). 현재 port는 xyz만 사용해서 호환됨.
- **히스토그램 로직**: `_compute_lidar_histogram` (dataset.py:173-210)은 NavSim의 `_get_lidar_feature` 내부 `splat_points`와 **수식·파라미터 모두 동일**. 바뀐 건 입력 포인트 출처뿐.
- 결과: 1채널(또는 ground plane 포함 시 2채널) 256×256 히스토그램 텐서.

### 2.3 좌표계 변환의 이유 (자주 실수하는 지점)

각 LiDAR sensor는 자기 sensor 좌표계의 점을 반환한다. 이걸 ego(차체) 좌표계로 끌어와야 6개 사이의 정렬이 맞고, BEV histogram에서 동일 격자에 splat된다. `calibrated_sensor`의 `rotation`(quaternion)·`translation`(meter)이 sensor→ego 변환을 정의한다.

> CAUTION: `pyquaternion.Quaternion.rotation_matrix`는 (x, y, z, w) 순이 아니라 **(w, x, y, z)** 순으로 인자를 받는다. TruckScenes JSON의 `rotation` 필드는 `[w, x, y, z]` 순서이므로 그대로 넘겨주면 된다 (dataset.py:160).

---

## 3. 데이터 파이프라인 — pickle 제거, devkit 직접 호출

NavSim은 `Scene.from_scene_dict_list()`로 사전 변환된 pickle을 읽고, `TransfuserFeatureBuilder` / `TransfuserTargetBuilder`로 feature·target을 분리해 gzip 캐시한다. TruckScenes는 mini~v1.1까지 규모가 크지 않으므로 단순화했다:

```python
# dataset.py:49-125 요약
class TruckScenesDataset(Dataset):
    def __init__(self, ts, config, num_future_samples=8, split_tokens=None):
        self._ts = ts
        self._config = config
        self._num_future_samples = num_future_samples
        self._sample_tokens = self._collect_valid_samples(split_tokens)

    def __getitem__(self, idx):
        sample = self._ts.get("sample", self._sample_tokens[idx])
        features = {
            "camera_feature": self._get_camera_feature(sample),
            "lidar_feature":  self._get_lidar_feature(sample),
            "status_feature": self._get_status_feature(sample),
        }
        targets = {
            "trajectory":    self._get_trajectory_target(sample),
            "agent_states":  ..., "agent_labels": ...,
        }
        return features, targets
```

- **장점**: 캐시·pickle 변환 생략 → 데이터 추가/수정 즉시 반영. 코드 흐름 단순.
- **단점**: 매 sample마다 4×이미지 디코딩 + 6×PCD 로딩이 일어나 디스크 I/O가 병목. v1.1 trainval 스케일에서 오랜 시간 학습 시 캐시 도입을 검토 필요 (`04-current-port-review.md` §4).

### 3.1 split_tokens
`split_tokens`는 scene token 리스트. None이면 모든 scene을 사용한다. **scene-level split**을 권장 — 같은 scene 내에서 frame을 train/val로 나누면 시간적으로 인접한 frame이 양쪽에 들어가 leakage가 생긴다.

### 3.2 sample 유효성
`_collect_valid_samples` (dataset.py:73-95): scene을 처음부터 끝까지 순회하면서 `next` 포인터가 `num_future_samples`개 더 있는 sample만 학습 대상으로. scene 끝의 마지막 8개 sample은 trajectory target을 못 만들기 때문에 제외.

---

## 4. Status feature 설계 (4D, ego_pose 미분 + chassis IMU)

NavSim은 `driving_command(4) + ego_velocity(2) + ego_acceleration(2) = 8D`를 사용하는데, TruckScenes에는 동등한 routing 신호가 없다. **그리고 chassis CAN bus의 vx/vy는 ~32% sample에서 0으로 stuck되는 데이터 품질 이슈가 있다** (측정: `tools/checks/check_chassis_zero_rate.py`, `tools/checks/check_cabin_vs_chassis.py`). 이 32% 중 91%는 실제로 움직이는 sample이라 학습 입력이 거짓말. 그래서:

```python
# dataset.py — _get_status_feature
chassis = ts.getclosest("ego_motion_chassis", sd["timestamp"])
ax, ay = float(chassis["ax"]), float(chassis["ay"])  # IMU는 신뢰 가능
vx, vy = self._estimate_ego_velocity(sample)         # ego_pose 미분
return torch.tensor([vx, vy, ax, ay], dtype=torch.float32)
```

`_estimate_ego_velocity`는 devkit `box_velocity` (truckscenes-devkit/.../truckscenes.py:443)와 동일 패턴 — **prev/next sample의 ego_pose translation centered difference**를 ego frame으로 회전:
- prev + next 모두 있으면 `(next - prev) / 2Δt` (centered diff)
- 한쪽만 있으면 forward/backward
- `max_time_diff = 1.5s` 가드 (벗어나면 0 fallback)

**왜 ego_pose 미분?**
- ego_pose는 GNSS/INS 기반이라 신뢰성 높음 (정차/주행 모두 정확)
- chassis CAN bus의 wheel-speed/GPS-velocity stream이 일부 시간 구간에 stuck → 신뢰 X
- devkit이 `box_velocity`에서 ann translation 미분으로 GT velocity를 만드는 것과 동일 사상 (외부 차량은 CAN bus 없으니 통일성 위해 ann 미분 사용)
- **대안**: `ego_motion_cabin`은 2.4%만 0이라 chassis보다 훨씬 정확. cabin으로 바꿔도 거의 같은 결과 (검증: `tools/checks/check_cabin_vs_chassis.py`).

**좌표계**: ego_pose 미분 결과를 현재 ego yaw inverse로 회전 → ego frame velocity. `ax/ay`는 chassis frame이지만 ego와 거의 일치 (회전 차이 미미).

**모델 차원**: `_status_encoding = nn.Linear(4, tf_d_model)`로 4D 그대로 (model.py).

> 향후 routing 신호를 줄 방법이 생기면 (e.g., HD map waypoint) status를 8D 또는 그 이상으로 확장 가능.

---

## 5. Trajectory Target 생성

```python
# dataset.py:221-266 요약
current_ego = ts.get("ego_pose", lidar_sd["ego_pose_token"])
current_pos = current_ego["translation"][:2]
current_yaw = _quaternion_to_yaw(Quaternion(current_ego["rotation"]))

trajectory = np.zeros((num_future_samples, 3))
next_token = sample.get("next", "")
for i in range(num_future_samples):
    if not next_token:
        if i > 0:
            trajectory[i] = trajectory[i-1]   # 끝에서는 마지막 pose 반복
        continue
    next_sample = ts.get("sample", next_token)
    next_ego = ts.get("ego_pose", ts.get("sample_data", next_sample["data"][lidar_channel])["ego_pose_token"])

    delta = future_pos - current_pos
    cos_yaw, sin_yaw = np.cos(-current_yaw), np.sin(-current_yaw)
    local_x = delta[0]*cos_yaw - delta[1]*sin_yaw
    local_y = delta[0]*sin_yaw + delta[1]*cos_yaw
    local_heading = (future_yaw - current_yaw + np.pi) % (2*np.pi) - np.pi
    trajectory[i] = [local_x, local_y, local_heading]
    next_token = next_sample.get("next", "")
```

- TruckScenes의 sample 간격은 **0.5s** (10 Hz keyframe). 따라서 NavSim의 `trajectory_sampling_interval=0.5`와 자연스럽게 맞아떨어진다 — 별도 보간 없이 sample chain만 따라가면 된다.
- **8 frames × 0.5s = 4.0s 미래 horizon** (config.py:87-89, evaluate.py:24의 `EVAL_HORIZONS=[1,2,3]`초와 호환).

---

## 6. Agent (Vehicle Detection) Target

```python
# dataset.py:268-317 요약
boxes = ts.get_boxes(sd["token"])         # global frame Box3D
ego_pose = ts.get("ego_pose", sd["ego_pose_token"])
ego_pos = np.array(ego_pose["translation"])
ego_rot = Quaternion(ego_pose["rotation"])

for box in boxes:
    if not _is_vehicle_category(box.name):     # vehicle.* prefix, ego_trailer 제외
        continue
    center = ego_rot.inverse.rotate(box.center - ego_pos)   # global → ego
    x, y = center[0], center[1]
    if not (lidar_min_x <= x <= lidar_max_x and lidar_min_y <= y <= lidar_max_y):
        continue
    box_yaw = _quaternion_to_yaw(box.orientation)
    heading = (box_yaw - ego_yaw + np.pi) % (2*np.pi) - np.pi
    length, width = box.wlh[1], box.wlh[0]    # wlh = [width, length, height]
    agent_states_list.append([x, y, heading, length, width])

# closest 30 by L2 distance, zero-padding
```

핵심 포인트:

- **카테고리 prefix 매칭**: TruckScenes는 `vehicle.car.compact`, `vehicle.truck.semitrailer` 등 계층형 라벨. NavSim의 `name == "vehicle"` 정확 일치는 0건이 된다 (`87cedbe` 커밋의 Fix 참고).
- **`ego_trailer` 제외**: 트럭 자기 트레일러는 ego의 일부이므로 detection target에서 제외.
- **wlh 인덱싱 주의**: `Box3D.wlh = [width, length, height]` 순서. 모델·loss는 BoundingBox2DIndex 순(`length, width`)이므로 `wlh[1] (length), wlh[0] (width)`로 풀어 넣어야 한다.
- **range filter**: BEV 영역(`±32m`) 안에 있는 box만. 안 그러면 멀리 있는 차들이 학습 분포를 흐린다.
- **zero-padding + label float**: NavSim은 label을 boolean으로 두지만 현재 port는 float(`1.0/0.0`)로 둔다 — loss 안에서 결국 float 변환되므로 결과는 동일.

---

## 7. BEV Semantic — 왜 비활성화?

NavSim의 BEV semantic target은 (1) HD map polygon/linestring을 ego 좌표계로 변환해 OpenCV로 라스터화, (2) box들을 같은 BEV 프레임에 fillPoly, 두 단계로 만들어진다 (`transfuser_features.py::_compute_bev_semantic_map`). HD map이 없는 TruckScenes에서는 (1)을 만들 수 없어 의미 있는 target을 생성할 수 없다. 그래서:

- `config.use_bev_semantic = False`, `bev_semantic_weight = 0.0` (config.py:55, 74)
- 모델의 `_bev_semantic_head`는 그대로 두되 `forward`에서 weight=0이면 head를 호출하지 않는다 (model.py:106).
- target dict에도 `bev_semantic_map` 키를 안 만든다 (dataset.py:119-123).

**향후 HD map이나 거친 semantic 정보가 생기면**:
1. `use_bev_semantic=True`, `bev_semantic_weight=10.0` (NavSim 값)
2. dataset에서 `bev_semantic_map` 키를 만들어 (H/2, W) 정수 텐서를 반환
3. 모델·loss는 코드 변경 없이 동작

---

## 8. 좌표계 컨벤션 정리 (가장 자주 실수하는 지점)

| 좌표계 | 정의 | 코드에서 어떻게 다루나 |
|---|---|---|
| **global** | TruckScenes 글로벌 origin (지도 평면) | `ego_pose.translation`, `box.center` 등 모두 global |
| **ego** | 현재 sample의 ego(차체) 중심 | dataset에서 `ego_rot.inverse.rotate(box.center - ego_pos)`로 변환 (dataset.py:288-289) |
| **sensor** | 각 LiDAR/카메라 자기 좌표계 | `calibrated_sensor`의 R, t로 sensor→ego 변환 (dataset.py:159-163) |
| **BEV pixel** | 256×256 격자, 픽셀 좌표 (i, j) | `(x, y)` (ego) → `((y - ymin)*ppm, (x - xmin)*ppm)` 식으로 splat. `splat_points`는 numpy `histogramdd`가 처리 |
| **chassis** | CAN bus의 차체 기준 (대부분 ego와 거의 일치) | status feature는 그대로 사용 (4D 그대로 모델에 입력) |

**자주 헷갈리는 부분**:
- `Quaternion.inverse.rotate(v)`로 global → local 변환. 반대로 `rotate(v)`는 local → global.
- yaw는 `_quaternion_to_yaw`로 (1,0,0)을 회전한 결과의 atan2로 추출. `[-π, π]` 범위.
- box의 yaw는 global frame이므로 ego에서 본 heading은 `box_yaw - ego_yaw`를 다시 `(-π, π]`로 wrap.

---

## 9. 요약 체크리스트 (다른 트럭/로봇 데이터셋으로 옮길 때)

다음을 항목별로 답할 수 있으면 적용 가능:

- [ ] 카메라 N대 → stitching 후 가로 해상도 결정 → `camera_width`, `img_horz_anchors` 동기화
- [ ] LiDAR M대 → 모두 ego frame으로 변환 후 BEV histogram 합치기. `pixels_per_meter`·범위는 그대로 두는 편이 안전
- [ ] CAN/IMU에서 `vx, vy, ax, ay` 추출 가능 여부 → `status_feature` 4D 채우기. 안 되면 zero padding
- [ ] routing/driving_command 가능 여부 → 8D로 확장할지 4D 유지할지 결정 (모델 `_status_encoding` Linear도 함께 수정)
- [ ] vehicle 카테고리 라벨링 규약 확인 → `_is_vehicle_category` 갱신
- [ ] sample 간격 ↔ `trajectory_sampling_interval` 일치 확인 (다르면 evaluate.py 수식도 같이 수정)
- [ ] HD map 보유 여부 → BEV semantic 활성화/비활성화 결정
- [ ] devkit이 `get_boxes`, `ego_pose`, `calibrated_sensor`, `ego_motion_chassis` 등 nuScenes-like API를 제공하는지 확인 (없으면 어댑터 작성)
