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

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyquaternion import Quaternion

from model.config import TransfuserConfig
from model.model import TransfuserModel
from model.enums import BoundingBox2DIndex
from dataset.dataset import (
    TruckScenesDataset,
    _is_vehicle_category,
    _get_reference_channel,
    _quaternion_to_yaw,
)


def _short_category(name: str) -> str:
    """vehicle.bus.bendy → bus, vehicle.car → car (시각화용 짧은 라벨)."""
    parts = name.split(".")
    return parts[1] if len(parts) >= 2 else name


def _get_gt_boxes_with_category(ts, sample_token, config):
    """현재 sample의 vehicle box를 ego frame에서 (x, y, h, L, W, category)로 반환.
    dataset.py의 _get_agent_targets와 동일한 필터/정렬 적용 — 카테고리 정보까지 포함."""
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
        x, y = float(center[0]), float(center[1])
        if not (config.lidar_min_x <= x <= config.lidar_max_x and
                config.lidar_min_y <= y <= config.lidar_max_y):
            continue
        box_yaw = _quaternion_to_yaw(box.orientation)
        h = (box_yaw - ego_yaw + np.pi) % (2 * np.pi) - np.pi
        length, width = float(box.wlh[1]), float(box.wlh[0])
        items.append((x, y, float(h), length, width, _short_category(box.name)))

    # 가까운 거리순으로 정렬해서 num_bounding_boxes만 유지 (dataset과 동일)
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
                   gt_categories=None):
    """한 샘플에 대해 camera + BEV 서브플롯 생성.
    gt_categories: agent_states 순서와 동일한 길이의 카테고리 리스트 (없으면 라벨 미표시)."""
    # BEV가 좁으면 카테고리 라벨이 안 보여서 ratio 조정 + 전체 fig 사이즈 키움
    fig, (ax_cam, ax_bev) = plt.subplots(1, 2, figsize=(24, 12),
                                          gridspec_kw={"width_ratios": [1.6, 1]})

    # === Camera (stitched 4-view) ===
    cam = features["camera_feature"].cpu().numpy()  # (3, H, W)
    cam_img = np.transpose(cam, (1, 2, 0))
    ax_cam.imshow(cam_img)
    ax_cam.set_title(f"Sample {sample_idx} — Camera (stitched)")
    ax_cam.axis("off")

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
        gt_categories = [it[5] for it in gt_items]

        out_path = out_dir / f"sample_{idx:05d}.png"
        _render_sample(idx, features, targets, predictions, config, out_path,
                       gt_categories=gt_categories)
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
