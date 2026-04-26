"""
시각화 — BEV + camera + GT trajectory/bboxes + (선택) 모델 prediction까지 오버레이해서 PNG로 저장.
학습 데이터가 좌표계/스케일상 말이 되는지 눈으로 확인 + checkpoint가 있으면 예측 품질도 같이 점검.

사용법:
    # checkpoint 없이 GT만 시각화
    python tools/visualize.py --dataroot <path> --num 5 --out viz/

    # checkpoint 로드해서 prediction까지
    python tools/visualize.py --dataroot <path> --checkpoint checkpoint_epoch10.pt --num 5
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyquaternion import Quaternion
from truckscenes.utils.geometry_utils import view_points

from model.config import TransfuserConfig
from model.model import TransfuserModel
from model.enums import BoundingBox2DIndex
from dataset.dataset import (
    TruckScenesDataset,
    CAMERA_CHANNELS,
    _is_vehicle_category,
    _get_reference_channel,
    _quaternion_to_yaw,
)


def _ego_to_camera(points_ego: np.ndarray, ts, sd_token: str):
    """Ego-frame (N,3) 점을 카메라 frame으로 변환 + 이미지로 projection.
    반환: (proj_uv: (N,2), depth: (N,)) — depth>0인 점만 화면 앞쪽."""
    sd = ts.get("sample_data", sd_token)
    cs = ts.get("calibrated_sensor", sd["calibrated_sensor_token"])
    K = np.array(cs["camera_intrinsic"])
    cam_t = np.array(cs["translation"])
    cam_R = Quaternion(cs["rotation"]).rotation_matrix  # ego←cam

    # ego→cam: p_cam = R^T (p_ego - t). row-vector form: (p_ego - t) @ R
    p_cam = (points_ego - cam_t) @ cam_R  # (N, 3)
    # view_points는 column-vector 입력
    proj = view_points(p_cam.T, K, normalize=True)  # (3, N) — u, v, 1
    return proj[:2].T, p_cam[:, 2]


def _box_3d_corners(x: float, y: float, heading: float, length: float, width: float,
                     z: float = 0.0, height: float = 1.7):
    """ego-frame box의 8 corner 3D 좌표 (N=8, 3). z는 box 중심의 ego-frame z 좌표."""
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    hl, hw, hh = length / 2, width / 2, height / 2
    # box-local corners (forward, left, up)
    corners_local = np.array([
        [+hl, +hw, +hh], [+hl, -hw, +hh], [-hl, -hw, +hh], [-hl, +hw, +hh],
        [+hl, +hw, -hh], [+hl, -hw, -hh], [-hl, -hw, -hh], [-hl, +hw, -hh],
    ])
    R = np.array([[cos_h, -sin_h, 0], [sin_h, cos_h, 0], [0, 0, 1]])
    corners_ego = corners_local @ R.T + np.array([x, y, z])
    return corners_ego


# 박스 corner 연결 순서 (앞면 4개, 뒷면 4개, 양쪽 연결 4개)
_BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # 윗면
    (4, 5), (5, 6), (6, 7), (7, 4),  # 아랫면
    (0, 4), (1, 5), (2, 6), (3, 7),  # 수직 기둥
]


def _draw_projected_box(ax, corners_ego, ts, sd_token, color, linewidth=1.5):
    """3D ego-frame box의 8 corner를 카메라에 투영 후 ax에 그림.
    뒤쪽이거나 일부 corner depth<=0이면 그려지지 않음."""
    uv, depth = _ego_to_camera(corners_ego, ts, sd_token)
    if (depth <= 0.5).any():  # 카메라 너무 가깝거나 뒤쪽이면 스킵
        return
    for i, j in _BOX_EDGES:
        ax.plot([uv[i, 0], uv[j, 0]], [uv[i, 1], uv[j, 1]],
                color=color, linewidth=linewidth, alpha=0.9)


def _draw_projected_traj(ax, traj_xy, ts, sd_token, color, marker="o"):
    """Trajectory points (N, 2) — x forward, y left. 지면(z=0) 가정으로 카메라에 투영."""
    if len(traj_xy) == 0:
        return
    pts_ego = np.column_stack([traj_xy, np.zeros(len(traj_xy))])  # z=0 (ground)
    uv, depth = _ego_to_camera(pts_ego, ts, sd_token)
    valid = depth > 0.5
    if not valid.any():
        return
    ax.plot(uv[valid, 0], uv[valid, 1], color=color, linewidth=2,
            marker=marker, markersize=4)


def _short_category(name: str) -> str:
    """vehicle.bus.bendy → bus, vehicle.car → car (시각화용 짧은 라벨)."""
    parts = name.split(".")
    return parts[1] if len(parts) >= 2 else name


def _get_gt_boxes_with_category(ts, sample_token, config):
    """현재 sample의 vehicle box를 ego frame에서 7-tuple로 반환:
    (x, y, z, heading, L, W, H, category).
    z와 H를 함께 반환하므로 카메라 projection 시 실제 사이즈로 그릴 수 있음.
    dataset.py의 _get_agent_targets와 동일한 필터/정렬 적용."""
    sample = ts.get("sample", sample_token)
    lidar_channel = _get_reference_channel(sample)
    sd = ts.get("sample_data", sample["data"][lidar_channel])
    boxes = ts.get_boxes(sd["token"])

    ego_pose = ts.get("ego_pose", sd["ego_pose_token"])
    ego_pos = np.array(ego_pose["translation"])
    ego_rot = Quaternion(ego_pose["rotation"])
    ego_yaw = _quaternion_to_yaw(ego_rot)

    items = []
    for box in boxes:
        if not _is_vehicle_category(box.name):
            continue
        center = box.center - ego_pos
        center = ego_rot.inverse.rotate(center)
        x, y, z = float(center[0]), float(center[1]), float(center[2])
        if not (config.lidar_min_x <= x <= config.lidar_max_x and
                config.lidar_min_y <= y <= config.lidar_max_y):
            continue
        box_yaw = _quaternion_to_yaw(box.orientation)
        h = (box_yaw - ego_yaw + np.pi) % (2 * np.pi) - np.pi
        # truckscenes wlh = [width, length, height]
        length = float(box.wlh[1])
        width = float(box.wlh[0])
        height = float(box.wlh[2])
        items.append((x, y, z, float(h), length, width, height,
                      _short_category(box.name)))

    items.sort(key=lambda it: it[0] ** 2 + it[1] ** 2)
    return items[: config.num_bounding_boxes]


def _draw_bbox(ax, x, y, heading, length, width, color, label=None):
    """BEV axis 위에 oriented bounding box 그리기."""
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    hl, hw = length / 2, width / 2
    corners = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
    rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    corners = corners @ rot.T + np.array([x, y])
    poly = mpatches.Polygon(corners, closed=True, edgecolor=color,
                             facecolor="none", linewidth=1.5, label=label)
    ax.add_patch(poly)


def _render_sample(sample_idx, features, targets, predictions, config, out_path,
                   gt_categories=None, ts=None, sample_token=None):
    """한 샘플에 대해 camera 4개 + BEV 서브플롯 생성.
    ts/sample_token이 주어지면 카메라마다 GT/pred 박스+trajectory를 image plane에 투영해서 그림.
    gt_categories: agent_states 순서와 동일한 길이의 카테고리 리스트 (없으면 라벨 미표시)."""
    # 위쪽: 카메라 4개 (1x4), 아래쪽: BEV (전체 너비)
    fig = plt.figure(figsize=(24, 14))
    gs = gridspec.GridSpec(2, 4, height_ratios=[1, 2.2], figure=fig, hspace=0.15)
    ax_cams = [fig.add_subplot(gs[0, i]) for i in range(4)]
    ax_bev = fig.add_subplot(gs[1, :])

    # === Cameras (4개, uncropped) + projection ===
    if ts is not None and sample_token is not None:
        sample_obj = ts.get("sample", sample_token)
        traj_gt_xy = targets["trajectory"].cpu().numpy()[:, :2]

        # GT 박스: 실제 ego-frame z, height을 가진 정확한 3D corner 사용
        # (agent_states는 5D라 H/z가 빠져 있어 카메라 투영엔 부정확)
        gt_items = _get_gt_boxes_with_category(ts, sample_token, config)
        gt_corners_list = [
            _box_3d_corners(x, y, h, L, W, z=z, height=H)
            for (x, y, z, h, L, W, H, _cat) in gt_items
        ]

        # Pred 박스: 모델은 (x,y,h,L,W)만 출력 → z/H는 GT 평균으로 fallback
        # (없으면 차량 평균 height=1.7m, 지면 위 z=H/2 사용)
        if gt_items:
            fallback_z = float(np.mean([it[2] for it in gt_items]))
            fallback_H = float(np.mean([it[6] for it in gt_items]))
        else:
            fallback_H = 1.7
            fallback_z = fallback_H / 2.0  # 지면 위에 놓인 차량

        pred_corners_list = []
        pred_scores = []
        pred_traj_xy = None
        if predictions is not None:
            pred_states = predictions["agent_states"][0].cpu().numpy()
            pred_logits = predictions["agent_labels"][0].cpu().numpy()
            for j in range(len(pred_logits)):
                if pred_logits[j] < 0:
                    continue
                px = pred_states[j, BoundingBox2DIndex.X]
                py = pred_states[j, BoundingBox2DIndex.Y]
                ph = pred_states[j, BoundingBox2DIndex.HEADING]
                pl = max(pred_states[j, BoundingBox2DIndex.LENGTH], 0.5)
                pw = max(pred_states[j, BoundingBox2DIndex.WIDTH], 0.5)
                pred_corners_list.append(
                    _box_3d_corners(px, py, ph, pl, pw, z=fallback_z, height=fallback_H)
                )
                pred_scores.append(float(1.0 / (1.0 + np.exp(-pred_logits[j]))))
            pred_traj_xy = predictions["trajectory"][0].cpu().numpy()[:, :2]

        for ax_cam, (channel, _) in zip(ax_cams, CAMERA_CHANNELS):
            sd = ts.get("sample_data", sample_obj["data"][channel])
            img_path = Path(ts.dataroot) / sd["filename"]
            img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            ax_cam.imshow(img)
            ax_cam.set_title(channel.replace("CAMERA_", ""), fontsize=10)
            ax_cam.set_xlim(0, img.shape[1])
            ax_cam.set_ylim(img.shape[0], 0)
            ax_cam.axis("off")

            for corners in gt_corners_list:
                _draw_projected_box(ax_cam, corners, ts, sd["token"],
                                    color="lime", linewidth=1.5)
            for corners in pred_corners_list:
                _draw_projected_box(ax_cam, corners, ts, sd["token"],
                                    color="red", linewidth=1.5)
            _draw_projected_traj(ax_cam, traj_gt_xy, ts, sd["token"],
                                 color="lime", marker="o")
            if pred_traj_xy is not None:
                _draw_projected_traj(ax_cam, pred_traj_xy, ts, sd["token"],
                                     color="red", marker="x")
    else:
        # ts 미제공 시 fallback: stitched feature image
        cam = features["camera_feature"].cpu().numpy()
        cam_img = np.transpose(cam, (1, 2, 0))
        ax_cams[0].imshow(cam_img)
        ax_cams[0].axis("off")
        for ax in ax_cams[1:]:
            ax.axis("off")

    # === BEV ===
    # LiDAR histogram을 배경으로
    lidar = features["lidar_feature"].cpu().numpy()  # (1 또는 2, H, W)
    # 첫 채널(또는 sum)을 sky view로 표시
    lidar_bg = lidar[0] if lidar.shape[0] == 1 else lidar.sum(0)

    # BEV 좌표 변환: ego frame (+x forward, +y LEFT) → plot frame (+up, +left)
    # 매핑: plot_x = -ego_y, plot_y = ego_x. 이는 CCW π/2 rotation이므로
    # 모든 heading도 plot에 그릴 땐 π/2 만큼 추가 회전해야 함.
    #
    # lidar_feature 매트릭스는 shape (C, H=x_bins, W=y_bins):
    #   - 축0 = ego_x (forward) → plot_y로 매핑
    #   - 축1 = ego_y (LEFT)   → plot_x로 매핑하되 -부호 (플롯 왼쪽이 +ego_y)
    # [:, ::-1]로 y_bins 반전 + origin="lower"로 row 0을 bottom에 두면 정렬 완료.
    extent = [config.lidar_min_y, config.lidar_max_y,
              config.lidar_min_x, config.lidar_max_x]
    ax_bev.imshow(lidar_bg[:, ::-1], extent=extent, cmap="gray_r",
                  origin="lower", alpha=0.6)

    # Ego vehicle — heading=0 + plot 회전 π/2 → 세로로 서있는 박스로 그려짐
    _draw_bbox(ax_bev, 0, 0, np.pi / 2, 6.9, 2.5, color="blue", label="ego")
    ax_bev.plot(0, 0, "b*", markersize=12)

    # GT trajectory
    traj_gt = targets["trajectory"].cpu().numpy()  # (num_poses, 3)
    # BEV에서 x-forward는 위쪽 축, y-left는 왼쪽 축
    # ax_bev의 x축 = ego y (left), y축 = ego x (forward)
    ax_bev.plot(-traj_gt[:, 1], traj_gt[:, 0], "g-o", markersize=4, linewidth=2, label="GT traj")

    # GT agent boxes — 각 박스 위에 카테고리 라벨 표시 (없으면 인덱스)
    gt_states = targets["agent_states"].cpu().numpy()
    gt_labels = targets["agent_labels"].cpu().numpy()
    first_valid = True
    for j in range(len(gt_labels)):
        if gt_labels[j] < 0.5:
            continue
        x, y, h, length, width = gt_states[j]
        _draw_bbox(ax_bev, -y, x, h + np.pi / 2, length, width,
                   color="green", label="GT agent" if first_valid else None)
        first_valid = False
        # 라벨: gt_categories가 있으면 카테고리(예: "car"), 없으면 인덱스
        text = gt_categories[j] if gt_categories and j < len(gt_categories) else f"{j}"
        ax_bev.text(-y, x, text, fontsize=9, color="green",
                    ha="center", va="center", fontweight="bold")

    # Predictions (있으면)
    if predictions is not None:
        pred_traj = predictions["trajectory"][0].cpu().numpy()
        ax_bev.plot(-pred_traj[:, 1], pred_traj[:, 0], "r--o", markersize=4,
                    linewidth=2, label="Pred traj")

        pred_states = predictions["agent_states"][0].cpu().numpy()
        pred_logits = predictions["agent_labels"][0].cpu().numpy()
        # logit > 0 (= confidence > 0.5)인 것만 표시 + 점수 라벨
        for j in range(len(pred_logits)):
            if pred_logits[j] < 0:
                continue
            x, y, h = pred_states[j, BoundingBox2DIndex.X], pred_states[j, BoundingBox2DIndex.Y], pred_states[j, BoundingBox2DIndex.HEADING]
            length = pred_states[j, BoundingBox2DIndex.LENGTH]
            width = pred_states[j, BoundingBox2DIndex.WIDTH]
            score = float(1.0 / (1.0 + np.exp(-pred_logits[j])))  # sigmoid
            _draw_bbox(ax_bev, -y, x, h + np.pi / 2, length, width,
                       color="red", label="Pred agent" if j == 0 else None)
            ax_bev.text(-y, x, f"{score:.2f}", fontsize=9, color="red",
                        ha="center", va="center", fontweight="bold")

    # Status 정보 (vx, vy, ax, ay)
    status = features["status_feature"].cpu().numpy()
    ax_bev.set_title(
        f"Sample {sample_idx} — BEV (ego-centric)\n"
        f"v=({status[0]:+.1f}, {status[1]:+.1f}) m/s, "
        f"a=({status[2]:+.1f}, {status[3]:+.1f}) m/s²"
    )
    ax_bev.set_xlabel("y (left →)")
    ax_bev.set_ylabel("x (forward ↑)")
    ax_bev.set_xlim(config.lidar_min_y, config.lidar_max_y)
    ax_bev.set_ylim(config.lidar_min_x, config.lidar_max_x)
    ax_bev.set_aspect("equal")
    ax_bev.grid(alpha=0.3)
    ax_bev.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = TransfuserConfig()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from truckscenes.truckscenes import TruckScenes
    from truckscenes.utils.splits import create_splits_scenes

    print(f"Loading TruckScenes {args.version}...")
    ts = TruckScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    splits = create_splits_scenes()
    split_names = set(splits[args.split])
    split_tokens = [s["token"] for s in ts.scene if s["name"] in split_names]
    dataset = TruckScenesDataset(
        ts=ts, config=config, num_future_samples=config.num_poses,
        split_tokens=split_tokens,
    )
    print(f"Split '{args.split}': {len(dataset)} samples")

    # 모델 로드 (checkpoint 주어진 경우)
    model = None
    if args.checkpoint:
        model = TransfuserModel(config=config).to(device)
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.eval()
        print(f"Loaded: {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    # 균등 간격으로 샘플 인덱스 선택
    step = max(len(dataset) // args.num, 1)
    indices = [i * step for i in range(args.num)]

    for idx in indices:
        features, targets = dataset[idx]
        predictions = None
        if model is not None:
            with torch.no_grad():
                features_batch = {k: v.unsqueeze(0).to(device) for k, v in features.items()}
                predictions = model(features_batch)

        # GT 카테고리 라벨링용 — sample token으로 raw box 다시 fetch
        sample_token = dataset._sample_tokens[idx]
        gt_items = _get_gt_boxes_with_category(ts, sample_token, config)
        gt_categories = [it[-1] for it in gt_items]

        out_path = out_dir / f"sample_{idx:05d}.png"
        _render_sample(idx, features, targets, predictions, config, out_path,
                       gt_categories=gt_categories,
                       ts=ts, sample_token=sample_token)
        print(f"  saved: {out_path}")

    print(f"\n{args.num} visualizations saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, required=True)
    parser.add_argument("--version", type=str, default="v1.1-trainval")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--num", type=int, default=5)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--out", type=str, default="viz")
    args = parser.parse_args()
    main(args)
