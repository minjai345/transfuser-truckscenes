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

from model.config import TransfuserConfig
from model.model import TransfuserModel
from model.enums import BoundingBox2DIndex
from dataset.dataset import TruckScenesDataset


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


def _render_sample(sample_idx, features, targets, predictions, config, out_path):
    """한 샘플에 대해 camera + BEV 서브플롯 생성."""
    fig, (ax_cam, ax_bev) = plt.subplots(1, 2, figsize=(20, 8),
                                          gridspec_kw={"width_ratios": [2, 1]})

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

    # BEV extent: ego frame에서 x는 forward, y는 left.
    # lidar_feature는 [C, H=256, W=256], lidar_min_x~max_x / lidar_min_y~max_y 매핑.
    # 이미지 상으로는 x축=lidar_y (좌우), y축=lidar_x (상하).
    # 여기서는 간단히 ego-centric 좌표계 (x: forward=up, y: left=left)로 그림.
    extent = [config.lidar_min_y, config.lidar_max_y,
              config.lidar_min_x, config.lidar_max_x]
    ax_bev.imshow(lidar_bg.T[::-1], extent=extent, cmap="gray_r", origin="upper", alpha=0.6)

    # Ego vehicle (원점)
    _draw_bbox(ax_bev, 0, 0, 0, 6.9, 2.5, color="blue", label="ego")
    ax_bev.plot(0, 0, "b*", markersize=12)

    # GT trajectory
    traj_gt = targets["trajectory"].cpu().numpy()  # (num_poses, 3)
    # BEV에서 x-forward는 위쪽 축, y-left는 왼쪽 축
    # ax_bev의 x축 = ego y (left), y축 = ego x (forward)
    ax_bev.plot(-traj_gt[:, 1], traj_gt[:, 0], "g-o", markersize=4, linewidth=2, label="GT traj")

    # GT agent boxes
    gt_states = targets["agent_states"].cpu().numpy()
    gt_labels = targets["agent_labels"].cpu().numpy()
    for j in range(len(gt_labels)):
        if gt_labels[j] < 0.5:
            continue
        x, y, h, length, width = gt_states[j]
        # Ego → BEV 변환: BEV의 x = -ego_y, BEV의 y = ego_x, heading도 반영
        _draw_bbox(ax_bev, -y, x, h + np.pi / 2, length, width,
                   color="green", label="GT agent" if j == 0 else None)

    # Predictions (있으면)
    if predictions is not None:
        pred_traj = predictions["trajectory"][0].cpu().numpy()
        ax_bev.plot(-pred_traj[:, 1], pred_traj[:, 0], "r--o", markersize=4,
                    linewidth=2, label="Pred traj")

        pred_states = predictions["agent_states"][0].cpu().numpy()
        pred_logits = predictions["agent_labels"][0].cpu().numpy()
        # 확률 > 0.5인 것만
        for j in range(len(pred_logits)):
            if pred_logits[j] < 0:  # logit 기준
                continue
            x, y, h = pred_states[j, BoundingBox2DIndex.X], pred_states[j, BoundingBox2DIndex.Y], pred_states[j, BoundingBox2DIndex.HEADING]
            length = pred_states[j, BoundingBox2DIndex.LENGTH]
            width = pred_states[j, BoundingBox2DIndex.WIDTH]
            _draw_bbox(ax_bev, -y, x, h + np.pi / 2, length, width,
                       color="red", label="Pred agent" if j == 0 else None)

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
    fig.savefig(out_path, dpi=120)
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

        out_path = out_dir / f"sample_{idx:05d}.png"
        _render_sample(idx, features, targets, predictions, config, out_path)
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
