# 01. TransFuser 아키텍처 개요

> 논문: *TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving* (PAMI 2023)
> 본 문서는 논문 §3 (Method)의 핵심을 정리하고, 후속 문서(`02-reference-implementations.md`, `03-truckscenes-adaptation.md`)에서 구현 차이를 다룰 때 공통 기준점이 되도록 작성했다.

---

## 0. 한 줄 요약

이미지(전방 다중 카메라)와 LiDAR(BEV pseudo-image)를 **여러 해상도에서 self-attention으로 융합**하고, 융합된 latent로부터 **점-대-점 navigation의 미래 waypoint와 주변 객체 detection**을 동시에 예측하는 imitation learning 모델이다.

---

## 1. 문제 정의 (논문 §3.1)

- Behavior Cloning (BC). Expert policy `π*`가 만든 (input X, action a) 쌍을 사용해 policy `π_θ`를 supervised 학습.
- Input `X`: 카메라 RGB + LiDAR 포인트 클라우드 (+ 선택적으로 ego speed, target_point 등의 측정값).
- Output `W`: ego 차량의 BEV 좌표계 미래 waypoint 시퀀스. 공식 CARLA는 `T=4` waypoints, NavSim은 `num_poses=8` poses(`x, y, heading`).
- Loss: 기본은 waypoint L1 + 다양한 보조 loss (detection, semantic, depth, BEV map). 보조 loss는 backbone에 강한 inductive bias를 부여해 imitation 성능을 끌어올리는 역할.

---

## 2. 입력 표현 (논문 §3.2)

### 2.1 LiDAR BEV pseudo-image
- LiDAR 포인트를 ego 기준 정사각 BEV 그리드에 투영해 2D pseudo-image 생성.
- 공식 CARLA: `32m × 32m` 영역, `0.125 m/pixel` → `256 × 256`. **2-bin histogram** (지면 위/아래) → 2채널 + 선택적 target_point 채널 = 최대 3채널.
- NavSim/현재 port: 동일하게 `±32m`, **`4 px/m`** → `256 × 256`. 기본은 1채널(`above` only), `use_ground_plane=True`이면 2채널.
- 핵심 함수: `splat_points()` (NavSim/현재 port 동일 로직). `np.histogramdd` 후 `hist_max_per_pixel`(=5)로 클리핑·정규화.

### 2.2 RGB 이미지
- 공식 CARLA: 전방 3-cam (정면 + 좌60° + 우60°), 각 960×480에서 320×160으로 crop, 가로 concat → `704×160`. 합쳐진 FOV ≈ 132°.
- NavSim: 3-cam (`cam_l0/f0/r0`) 가로 concat → `1024×256`.
- 현재 port: 4-cam(LEFT_FRONT, RIGHT_FRONT, LEFT_BACK, RIGHT_BACK) → `1536×256` (`03-truckscenes-adaptation.md` 참조).

### 2.3 보조 측정값
- 공식 CARLA: ego speed scalar + 글로벌 target_point(루트 플래너로부터). Target은 (a) 2D vector로 GRU 입력에 concat, (b) BEV에 Gaussian으로 라스터화해 LiDAR 입력 채널 추가, 두 가지 형태로 사용.
- NavSim: `driving_command(4 one-hot) + ego_velocity(2) + ego_acceleration(2)` = **8D status vector**, transformer decoder의 keyval 토큰에 추가.
- 현재 port: NavSim과 동일한 keyval-token 방식. `driving_command`만 제거하고 4D(vel + acc).

---

## 3. Multi-scale Fusion Transformer (논문 §3.3, Fig. 2)

### 3.1 골격
- 이미지 인코더와 LiDAR 인코더를 **병렬**로 두고, 두 인코더의 **각 stage 직후 (총 4회)**에 GPT-style transformer로 토큰을 합쳐 정보를 교환한다.
- 병렬 구조이므로 각 인코더는 자기 modality에 특화된 hierarchical feature를 유지하면서, 동시에 다른 modality에서 관련 정보를 attention으로 끌어올 수 있다.

### 3.2 한 stage의 처리 (현재 port 기준 — backbone.py:201-226)
```text
image_feature  ──┐
                 ├── avgpool (anchor grid 8x48 / 8x8) ──┐
lidar_feature  ──┘                                       │
                                                         ├── concat 토큰 → GPT (n_layer=2, n_head=4)
                       위치 임베딩(pos_emb) + dropout    ┘   ↓
                                                  fused img/lidar 토큰을 분리
                                                          ↓
                                              원래 H,W로 bilinear upsample
                                                          ↓
                                           원본 feature에 residual add
```
- GPT 블록 (backbone.py:229-307): 토큰 = `[image_tokens; lidar_tokens]`. Self-attention만 수행하고 LayerNorm + MLP(`block_exp=4`)로 마무리. **Cross-attention이 아닌 self-attention**으로 구현되어 있음에 주의 — 두 modality가 한 시퀀스에 묶여 있어 자연스럽게 cross-modal context가 형성된다.
- 채널 수 차이 보정: image와 lidar의 channel이 다르면 `lidar_channel_to_img` / `img_channel_to_lidar` (1×1 conv)로 맞춰준다.
- 4-stage 모두 동일 구조, 단 `n_embd`만 stage 별 backbone feature width에 따라 변동.

### 3.3 융합 후 두 가지 출력 경로
- **Decoder 입력용 평탄화**: 마지막 stage의 lidar feature(또는 fused vector)를 `BEV feature grid`로 사용. 현재 port에서는 `bev_feature` (FPN top-down 적용 전, 8×8) → `_bev_downscale` Conv1×1 → 64 토큰으로 펼쳐 디코더의 keyval에 투입 (model.py:91-98).
- **FPN top-down 경로** (backbone.py:140-144): 마지막 lidar feature를 1×1 Conv로 64ch로 줄이고 upsample해서 BEV semantic / detection head 입력(8×8 → 16×16 → 32×32)을 만든다. 현재 port는 detection은 query head에서 하므로 이 경로는 BEV semantic이 켜져야 활성화.

---

## 4. 출력 head 디자인 — 논문 / 공식 / NavSim 차이

논문에서 가장 흔히 인용되는 그림은 공식 CARLA 구현 기준이지만, NavSim은 **task가 다르기 때문에 head를 단순화**했다. 셋의 head 구성 차이가 가장 큰 분기점이다.

### 4.1 공식 CARLA: GRU 기반 autoregressive 회귀
- **Waypoint GRU** (model.py:611-646): 2D 좌표를 한 step씩 풀어내는 autoregressive GRU. hidden_size=64, 입력 dim=`{2 (x,y) | 4 (x,y,target_x,target_y)}`. 4개의 미래 waypoint를 시간순으로 생성.
- **CenterNet detection head** (model.py:34-99): heatmap(GaussianFocal), wh, offset, yaw_class(12-bin) + yaw_res, velocity, brake. CenterNet 식 detection으로 box 후보를 BEV에서 직접 뽑음.
- **BEV segmentation head** (3-class), **Depth decoder**, **Semantic segmentation decoder** 추가.
- 보조 loss 11개를 `detailed_losses_weights`로 가중합 (config.py:134-136).

### 4.2 NavSim: query-based detection + trajectory
NavSim은 nuPlan/CARLA 같은 closed-loop driving이 아니라 **planning 출력만 평가**(trajectory 8 poses)하는 벤치마크라서, 보조 head를 대거 정리하고 query-based 디코더 구조로 단순화했다.
- **TrajectoryHead** (model.py:141-157): single trajectory query → MLP로 `num_poses × (x, y, heading)` 예측. autoregressive 아님.
- **AgentHead** (model.py:117-138): `num_bounding_boxes=30`개 query. 각각 (x, y, heading, length, width) 5-tuple + confidence logit. **Hungarian matching loss** (loss.py:54).
- **BEV semantic head** (model.py:32-58): HD map + box → 7-class semantic map 분할. CrossEntropy.
- 보조 head는 **Trajectory + Agent + BEV semantic**, 3개로 정리. depth/semantic은 끔(`use_depth=False`, `use_semantic=False` 기본값).

### 4.3 현재 TruckScenes port: NavSim 동일 + BEV semantic 비활성화
- TrajectoryHead, AgentHead는 NavSim과 거의 동일.
- BEV semantic head 자체는 모델 구조에 그대로 남아 있으나, HD map이 없으니 target도 안 만들고 loss weight=0이라 학습되지 않는다 (model.py:106, loss.py:22-24, config.py:74).

---

## 5. Loss

| 구성 | 공식 CARLA | NavSim / 현재 port |
|---|---|---|
| Trajectory | L1 (`loss_wp`, weight 1.0) | L1 (`config.trajectory_weight=10.0`) |
| Detection | CenterNet 5-loss(heatmap/wh/offset/yaw_cls/yaw_res), 각 weight 0.2 | Hungarian matching → BCE class + L1 box (weights 10:1) |
| BEV segmentation | CrossEntropy(3-class), weight 1.0 | CrossEntropy(7-class), weight 10.0 (현재 port에선 weight=0) |
| Depth | L1, weight 10.0 (multitask 옵션) | 사용 안 함 |
| Semantic seg | CrossEntropy, weight 1.0 (multitask 옵션) | 사용 안 함 |

→ 현재 port에서 실제로 활성화된 loss는 **trajectory L1 + agent BCE + agent L1** 세 개뿐 (loss.py:11-26).

---

## 6. 학습 측면 핵심 포인트 (논문 §3.4 / 공식 학습 설정)

- **Optimizer**: 공식은 AdamW. NavSim·현재 port는 Adam (`train.py:161`).
- **Learning rate**: 공식 `1e-4` 기본값, `--schedule 1`이면 epoch 30, 40에서 ×0.1 → ×0.1 (train.py:42-46, 190-199).
- **Augmentation**: 공식은 LiDAR/이미지 동기 회전 ±20°(`aug_max_rotation`), 90% 확률(`1 - inv_augment_prob`). NavSim·현재 port는 augmentation 없음.
- **Pretrained weights**: 이미지 인코더는 ImageNet pretrained, LiDAR 인코더는 from-scratch.
- **Batch size**: 공식 12/GPU(DDP), NavSim 미공개, 현재 port 4 (train.py:337).

---

## 7. 모델 아키텍처를 한 페이지에 요약

```
   ┌───────────────────────┐         ┌─────────────────────────┐
   │ Multi-camera RGB       │         │ Multi-LiDAR points (6)   │
   │ (현재 port 1536x256, 4) │         │ → ego frame 변환·합치기   │
   └─────────┬─────────────┘         │ → BEV histogram 1ch 256² │
             │                       └────────┬────────────────┘
             ▼                                ▼
     ImageEncoder (ResNet34, ImNet)     LidarEncoder (ResNet34, scratch)
             │  4 stage feature           │  4 stage feature
             │                              │
        ┌────┴──────────────────────────────┴────┐
        │  4-stage GPT Fusion (per stage):          │
        │  avgpool → concat → self-attn → split →   │
        │  upsample → residual add                  │
        └────┬─────────────────────────────────────┘
             ▼
     LiDAR final feature (8x8x512) → 1×1 conv (256ch)
                                       │
                       ┌───────────────┴────────────────┐
                       ▼                                ▼
              keyval (64 token) +              FPN top-down (선택)
              status_encoding (1 token)            │
                       │                            ▼
                       ▼                BEV semantic head (현재 port disabled)
            Transformer decoder (3-layer, 8-head, 256d)
                       │
            queries: 1 trajectory + 30 agent
                       │
        ┌──────────────┴────────────┐
        ▼                           ▼
TrajectoryHead              AgentHead
 (8 poses x 3)               (30 boxes x 5 + 30 logits)
```

---

## 참조

- 논문 PDF: `/home/minjai/projects/transfuser/TransFuser: Imitation with Transformer-Based.pdf`
- 논문↔공식 코드 매핑: `/home/minjai/projects/transfuser_paper_code_mapping.md` (다음 문서 `02`에서 자주 인용)
- 공식 구현: `/home/minjai/projects/transfuser/team_code_transfuser/`
- 현재 port: `/home/minjai/projects/transfuser-truckscenes/model/`
