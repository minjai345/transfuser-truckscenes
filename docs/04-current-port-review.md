# 04. 현재 Port 검토 + 개선 권고

> `02-reference-implementations.md`에서 셋의 차이를 정리했다면, 이 문서는 **`/home/minjai/projects/transfuser-truckscenes/`의 코드 한 줄 한 줄이 NavSim 원본과 잘 맞는가, 의도적 일탈은 정당한가, 빠진 것은 무엇인가**를 점검한 결과다. 권고 항목은 severity(HIGH / MEDIUM / LOW)와 file:line·수정 방향으로 정리.

---

## 1. 파일 인벤토리

| 경로 | 행 수 | 역할 |
|---|---|---|
| `model/backbone.py` | 353 | TransfuserBackbone (이미지·LiDAR 인코더 + 4-stage GPT) |
| `model/model.py` | 157 | TransfuserModel (decoder + heads) |
| `model/loss.py` | 98 | trajectory + Hungarian agent loss |
| `model/config.py` | 103 | TransfuserConfig dataclass |
| `model/enums.py` | 97 | StateSE2Index, BoundingBox2DIndex |
| `dataset/dataset.py` | 364 | TruckScenesDataset (4-cam stitch + 6-LiDAR merge + targets) |
| `train.py` | 362 | 학습 루프, optimizer, wandb 로깅, 체크포인트 |
| `evaluate.py` | 282 | L2 / collision @ 1/2/3s 메트릭 |
| `tools/overfit_test.py` | — | 1 batch overfit sanity check |
| `tools/visualize.py` | — | BEV/카메라 visualization |
| `tools/predict_video.py` | — | scene 단위 예측 비디오 생성 |
| `tools/data_stats.py` | — | 데이터셋 통계 |

---

## 2. NavSim 정합성 점검표

NavSim 원본과 줄 단위로 어떤 부분이 같은지·다른지 정리. 같으면 ✅, 의도적 다름은 ⚠️ + 근거, 잘못된 다름은 ❌.

### 2.1 model/model.py

| 항목 | NavSim | 현재 port (file:line) | 상태 |
|---|---|---|---|
| `_query_splits = [1, num_bounding_boxes]` | 동일 | model.py:18-21 | ✅ |
| `_keyval_embedding = nn.Embedding(8**2 + 1, tf_d_model)` | 동일 | model.py:26 | ✅ |
| `_query_embedding = nn.Embedding(sum(_query_splits), tf_d_model)` | 동일 | model.py:27 | ✅ |
| `_bev_downscale = nn.Conv2d(512, tf_d_model, 1)` | 동일 | model.py:29 | ✅ |
| `_status_encoding = nn.Linear(4+2+2, tf_d_model)` | 8D 입력 | model.py:30 (`Linear(4, ...)`) | ⚠️ TruckScenes에는 driving_command 없음 → 4D로 축소 (`03-truckscenes-adaptation.md` §4) |
| `_bev_semantic_head` (Conv-Conv-Upsample) | 동일 | model.py:32-58 | ✅ 구조 유지 (학습 시엔 weight=0이라 unused) |
| Transformer decoder (3-layer, 8-head, ffn=1024) | 동일 | model.py:60-68, config.py:61-65 | ✅ |
| `AgentHead` MLP states + label | 동일 | model.py:117-138 | ✅ |
| `TrajectoryHead` MLP num_poses × 3 | 동일 | model.py:141-157 | ✅ |
| BEV semantic head 호출 조건 | `if config.use_bev_semantic` | `if config.bev_semantic_weight > 0` (model.py:106) | ⚠️ 조건이 weight 기반으로 바뀜. 효과 동일하지만 일관성을 위해 NavSim과 같이 `use_bev_semantic` 플래그를 보는 게 더 명확. **LOW** |

### 2.2 model/loss.py

| 항목 | NavSim | 현재 port | 상태 |
|---|---|---|---|
| L1 trajectory loss | `F.l1_loss(predictions["trajectory"], targets["trajectory"])` | loss.py:12 | ✅ |
| Hungarian matching | `linear_sum_assignment(c)` | loss.py:54 | ✅ |
| `_get_ce_cost` 식 | (간소화된 BCE cost) | loss.py:74-83 | ✅ |
| `_get_l1_cost` (xy만) | xy만 | loss.py:86-92 | ✅ |
| latent_rad_thresh 필터 | `config.latent`일 때 적용 | loss.py:33-42 | ✅ (현재 port는 latent=False라 미작동) |
| BEV semantic loss 분기 | `Σ + bev_semantic_weight * CE` 항상 합산 | `if weight>0 and key in targets:` 조건부 (loss.py:22-24) | ⚠️ 현재 port는 target dict에 키가 없으면 안전하게 skip — 데이터 파이프라인이 BEV target을 안 만들기 때문에 필요한 가드. NavSim과 의도가 같음 |
| `num_gt_instances = max(1, ...)` 0-divide 방지 | 동일 | loss.py:45-46 | ✅ |

### 2.3 model/config.py

| 항목 | NavSim | 현재 port | 상태 |
|---|---|---|---|
| `image_architecture = "resnet34"`, `lidar_architecture = "resnet34"` | 동일 | config.py:11-12 | ✅ |
| `pixels_per_meter = 4.0`, `lidar_min/max ±32` | 동일 | config.py:18, 21-24 | ✅ |
| `camera_width=1024, camera_height=256` | 1024 | **1536** (config.py:31) | ⚠️ 4-cam stitching에 맞춰 확장 |
| `img_horz_anchors = camera_width // 32` | 32 | **48** (config.py:37) | ⚠️ 위와 동기화 — 정확히 맞음 |
| `n_layer=2, n_head=4, block_exp=4` | 동일 | config.py:41-43 | ✅ |
| `use_bev_semantic = True` (NavSim 기본) | True | **False** (config.py:55) | ⚠️ HD map 없음 |
| `bev_semantic_weight = 10.0` | 10.0 | **0.0** (config.py:74) | ⚠️ 위와 짝 |
| `num_bounding_boxes = 30`, `num_poses = 8` | 동일 | config.py:68, 87 | ✅ |
| `trajectory_weight=10.0, agent_class_weight=10.0, agent_box_weight=1.0` | 동일 | config.py:71-73 | ✅ |
| `trajectory_sampling_time/interval = 4.0/0.5` | nuPlan TrajectorySampling | config.py:88-89 (직접 보유) | ⚠️ nuPlan 의존성 제거 — 단순 필드로 보유 |

### 2.4 model/backbone.py

NavSim의 `transfuser_backbone.py`와 거의 동일. 주의 항목:

- **`lidar_time_frames = [1, 1, 1, 1]` 하드코딩** (backbone.py:54). NavSim도 같은 값이지만, 향후 multi-frame LiDAR fusion을 시도하려면 config 필드로 빼야 한다. **LOW** 권고 항목.
- 4-stage GPT 구조, anchor grid avgpool, 채널 변환(`lidar_channel_to_img`/`img_channel_to_lidar`), FPN top-down 모두 NavSim과 동일.

### 2.5 dataset/dataset.py

| 단계 | NavSim 대응 코드 | 현재 port | 비고 |
|---|---|---|---|
| 카메라 stitch | `_get_camera_feature` (3-cam, 1024×256) | 4-cam, 1.5:1 directional crop, 1536×256 (dataset.py:127-143) | TruckScenes 매핑 |
| LiDAR splat | `splat_points()` | `_compute_lidar_histogram` (dataset.py:173-210) | 식 동일 |
| Status | 8D concat | 4D (CAN bus) (dataset.py:212-219) | driving_command 제거 |
| Trajectory | `Scene.get_future_trajectory().poses` | sample chain + quaternion (dataset.py:221-266) | 직접 구현 |
| Agent | `name == "vehicle"` + `_xy_in_lidar` | prefix `vehicle.*` + ego_trailer 제외 + range (dataset.py:268-317) | 라벨 규약 차이 |
| BEV semantic | HD map polygon/linestring + boxes | 미생성 | HD map 없음 |

`__getitem__` 반환 구조는 NavSim과 동일한 `(features dict, targets dict)` 쌍이라, 모델·loss는 그대로 사용 가능.

### 2.6 train.py / evaluate.py

NavSim은 자체 `TransfuserAgent`가 학습 루프와 callback을 담당하지만 현재 port는 직접 PyTorch 학습 루프로 간소화했다.

| 항목 | NavSim 또는 일반 권장 | 현재 port | 비고 |
|---|---|---|---|
| Optimizer | Adam (`get_optimizers()`) | `torch.optim.Adam(lr=1e-4)` (train.py:161) | 동일 |
| LR scheduler | NavSim 없음, 공식은 step decay | `CosineAnnealingLR` (train.py:162) | 합리적 추가 |
| Gradient clip | 보통 1.0 | `max_norm=1.0` (train.py:188) | OK |
| Loss aggregation | callback 통해 자동 | 직접 `transfuser_loss` (train.py:183) | OK |
| 평가 메트릭 | NavSim PDM-score | L2 + collision @ 1/2/3s (evaluate.py) | 자체 정의 — `evaluate.py:136`에 `interval=0.5s` 가정 하드코딩 (LOW) |

---

## 3. 의도적 일탈의 정당성

| 일탈 | 정당성 |
|---|---|
| **Status 4D vs 8D** | TruckScenes에 routing/driving_command가 없고, 임의 zero-padding은 분포만 흐린다. 모델 차원도 `Linear(4, ...)`로 함께 줄였으므로 일관됨. ✅ |
| **camera_width 1024→1536** | 4-cam stitching에 맞춰 자연스럽게 확장. `img_horz_anchors`도 동기화. ✅ |
| **BEV semantic 비활성화** | HD map 부재 시의 합리적 선택. head는 살아있어 데이터 갖춰지면 즉시 활성화 가능. ✅ |
| **6-LiDAR ego 변환 합치기** | TruckScenes 센서 구성 차이 흡수. 합친 뒤 BEV 표현은 NavSim과 동일. ✅ |
| **vehicle prefix 매칭** | TruckScenes 카테고리 규약상 필수. ego_trailer 제외도 정확. ✅ |

---

## 4. 발견된 갭과 개선 권고

각 항목 형식: **(severity) · 근거 file:line · 수정 위치 · 수정 방향**.

### 4.1 [REJECTED] 이미지 정규화 — NavSim/CARLA 표준에 어긋남

- **이전 권고**: ImageNet mean/std 정규화 추가 (backbone이 pretrained라 분포 일치).
- **검증 결과**: NavSim transfuser와 공식 CARLA transfuser **둘 다 정규화 안 함** (검증: `tools/checks/check_devkit_render.py` 작업 중 NavSim raw 코드 확인).
  - NavSim `transfuser_features.py:72`: `tensor_image = transforms.ToTensor()(resized_image)` — [0,1] 범위만.
  - 공식 CARLA `data.py`: ToTensor도 없이 raw uint8 [0,255] 그대로 backbone 입력.
- **결정**: baseline reproducibility 우선 → **정규화 추가 안 함**. NavSim 패턴 그대로 ToTensor만.
- 일반 ML 모범 사례와는 다르지만 reference 구현이 그러하니 따라감. 향후 ablation으로 정규화 추가 효과 측정 가능.

### 4.1' [FIXED] Status feature의 chassis vx/vy 데이터 품질 이슈 — ego_pose 미분으로 교체

- **근거**: `tools/checks/check_chassis_zero_rate.py` 측정 결과 — train+val 23,902 sample 중 35.3%가 `chassis.vx==0 && chassis.vy==0`. **그중 91%는 실제로 1 m/s 이상 움직이는 sample** (ego_pose 미분으로 검증). 즉 32% sample이 가짜 0 status로 학습 입력에 들어가고 있었다.
- **검증 케이스**: idx=798 (val) — trajectory[0]에서 추정한 vx=23.6 m/s (고속도로 주행)인데 chassis.vx=0.000.
- **원인 추정**: TruckScenes의 `ego_motion_chassis` table에서 wheel-speed 또는 GPS-velocity stream이 일부 시간 구간 stuck (ax/ay는 IMU 기반이라 정상). 데이터셋 자체 quality 이슈로 우리가 못 고침.
- **수정** (반영 완료, dataset.py): `_get_status_feature`가 `_estimate_ego_velocity` 호출 — **prev/next sample의 ego_pose centered difference**로 vx/vy 직접 계산. devkit `box_velocity`(truckscenes.py:443)와 동일 사상. `ax/ay`는 chassis IMU 그대로.
- **검증**: idx=798에서 새 status `vx = 23.603 m/s` (정확). idx=0 (저속) `vx = 2.712 m/s` (chassis 2.594와 거의 일치). idx=1596 (정차) `vx = 0.000` ✓.
- **대안**: `ego_motion_cabin`은 2.4%만 0이라 chassis보다 훨씬 정확. cabin으로 교체해도 거의 같은 결과 (`tools/checks/check_cabin_vs_chassis.py`). 현재는 ego_pose 미분 방식 채택 (devkit pattern과 일치).

### 4.2 [MEDIUM] 데이터 augmentation 부재

- **근거**: dataset.py 어디에도 LiDAR 회전/이동 augmentation, 이미지 color jitter, 좌우 flip 등 없음. 공식 CARLA `data.py:214-220`는 `aug_max_rotation=20°` rotation을 90% 확률로 적용.
- **문제**: mini 데이터셋에서는 큰 차이 없지만, v1.1 trainval로 확장할 때 일반화 성능 차이가 커질 수 있다.
- **수정 위치**: `dataset.py` 전체적으로 augmentation hook 추가, config에 `aug_max_rotation`, `inv_augment_prob` 필드 추가.
- **수정 방향**:
  - LiDAR: 포인트 합친 후 ego 좌표계에서 z축 ±θ 회전 + 작은 translation. **trajectory target과 agent target도 같은 변환을 받아야 함**.
  - 카메라: PIL/torchvision의 `ColorJitter(brightness, contrast, saturation, hue)`. 좌우 flip은 4-cam 구성상 애매하므로 보류.
- **주의**: trajectory target 자체가 ego 좌표계에 있으므로, LiDAR rotation을 적용하면 trajectory도 같이 회전해야 일관됨. 별도 헬퍼 함수로 변환 정합성 확보.

### 4.3 [LOW] `lidar_time_frames` 하드코딩

- **근거**: backbone.py:54 `lidar_time_frames = [1, 1, 1, 1]` 하드코딩.
- **문제**: 향후 multi-frame LiDAR fusion(시간축 stacking)을 실험하려면 4 stage 각각 다른 time frame 수를 줄 수 있어야 하지만 현재는 코드 수정 필요.
- **수정 위치**: `model/config.py`에 필드 추가, `backbone.py:54`에서 그 값을 읽기.
- **수정 방향**:
  ```python
  # config.py
  lidar_time_frames: Tuple[int, ...] = (1, 1, 1, 1)
  # backbone.py
  lidar_time_frames = list(config.lidar_time_frames)
  ```

### 4.4 [LOW] evaluate.py의 sampling interval 하드코딩

- **근거**: evaluate.py:136 부근에서 1s/2s/3s를 trajectory 인덱스로 변환할 때 `int(horizon / 0.5) - 1` 식이 들어감.
- **문제**: `config.trajectory_sampling_interval`이 0.5s가 아니면 잘못된 인덱스를 보게 됨.
- **수정 위치**: `evaluate.py`에서 `idx = int(horizon / config.trajectory_sampling_interval) - 1`로 교체.

### 4.5 [LOW] 매번 디스크 로딩 — 캐시 옵션 부재

- **근거**: dataset.py:127-171 — sample마다 4 카메라 디코딩 + 6 LiDAR PCD 디코딩 발생.
- **문제**: v1.1 trainval(수만 샘플)로 확장 시 디스크 I/O가 GPU를 굶긴다.
- **수정 위치**: `dataset.py`에 optional 디스크 캐시(npz) 또는 LMDB 옵션 추가. 또는 `num_workers`를 충분히 키우고 prefetch.
- **수정 방향(가벼운 안)**: feature(camera tensor, lidar histogram, status)와 target을 한 번 계산해 `${cache_dir}/<sample_token>.pt`로 저장하고, `__getitem__`에서 존재하면 로드만. mini에서는 불필요.

### 4.6 [LOW] BEV semantic 호출 조건의 표현

- **근거**: model.py:106 `if self._config.bev_semantic_weight > 0:` 으로 head 호출 결정.
- **문제**: NavSim은 `use_bev_semantic` 플래그로 분기. 현재 코드는 weight=0이면 자연스럽게 비활성화되지만, 의도가 헷갈릴 수 있음.
- **수정 방향**: `if self._config.use_bev_semantic:` 으로 바꾸고 `bev_semantic_weight`는 loss 가중치로만 사용하도록 분리.

### 4.7 [LOW] 카테고리 prefix 매칭 자세히 — 결과 검증

- **근거**: dataset.py:45-46 `_is_vehicle_category` — `vehicle.*` 매칭, `ego_trailer` 제외.
- **확인 필요**: 실제 v1.1-mini에서 어떤 vehicle 서브카테고리가 등장하는지 `tools/data_stats.py`로 출력. 예외 케이스(예: `vehicle.bicycle` 같은 이종 객체)가 vehicle로 잡혀 있다면 detection target에서 빼는 게 더 적절할 수 있음.

---

## 5. 향후 확장 옵션

### 5.1 BEV semantic 재활성화
1. 외부에서 HD map 또는 거친 lane mask를 확보 (예: ego pose 기반 OSM 추출 + ego 좌표계 변환).
2. `dataset.py`에 `_get_bev_semantic_target(sample)` 추가, NavSim의 `_compute_bev_semantic_map`을 참고해 OpenCV로 라스터화.
3. `config.use_bev_semantic = True`, `bev_semantic_weight = 10.0`.
4. 모델·loss 코드 수정 불필요(이미 분기 존재).

### 5.2 Multi-frame LiDAR
- §4.3 권고를 적용한 후, dataset에서 직전 1~2 keyframe의 LiDAR도 합쳐 시간축 stacking 텐서를 만들고 `lidar_seq_len`을 키운다.
- backbone의 LiDAR encoder `in_chans`가 자동으로 늘어나도록 이미 코드가 되어 있음 (backbone.py:26-29).

### 5.3 Detection 보조 head 추가
- 공식 CARLA처럼 BEV CenterNet detection head를 query-based 위에 보조로 추가하면 backbone에 더 강한 inductive bias를 줄 수 있음.
- 다만 단순 도로 trajectory 정확도 향상에는 비용 대비 효과가 작을 가능성이 높음.

### 5.4 Pretrained weight 로드
- NavSim의 학습된 체크포인트는 nuPlan 포맷이라 직접 호환되지 않음. 백본(resnet34)만 timm pretrained로 가져오고 그 외는 from scratch 학습이 현실적.
