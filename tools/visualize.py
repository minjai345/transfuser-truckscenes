"""
시각화 — BEV + camera + GT trajectory/bboxes + (선택) 모델 prediction까지 오버레이해서 PNG로 저장.

박스 카메라 투영은 truckscenes-devkit 방식(`box.render(ax, view=K)`)을 그대로 사용해
yaw 외 pitch/roll까지 포함된 정확한 3D 회전이 적용됨.

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
from pyquaternion import Quaternion

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from truckscenes.utils.data_classes import Box
from truckscenes.utils.geometry_utils import (
    BoxVisibility,
    box_in_image,
    view_points,
)

from configs import TransfuserConfig, load_config
from model.model import TransfuserModel
from model.enums import BoundingBox2DIndex
from dataset.dataset import (
    TruckScenesDataset,
    CAMERA_CHANNELS,
    _is_vehicle_category,
    _get_reference_channel,
    _quaternion_to_yaw,
)


def _short_category(name: str) -> str:
    """vehicle.bus.bendy → bus, vehicle.car → car (시각화용 짧은 라벨)."""
    parts = name.split(".")
    return parts[1] if len(parts) >= 2 else name


def _get_gt_boxes_with_category(ts, sample_token, config):
    """현재 sample의 vehicle box를 ego frame에서 8-tuple로 반환:
    (x, y, z, heading, L, W, H, category).
    BEV 전용. 카메라 투영은 _draw_camera_devkit이 별도로 처리(devkit 사용).

    dataset.py의 _get_agent_targets와 동일한 필터/정렬 적용.
    """
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


def _ego_frame_box_to_sensor(box_ego: Box, cam_cs):
    """ego frame Box 객체를 카메라 sensor frame으로 in-place 변환.
    (예측 박스용 — 모델은 ego frame에서 (x, y, h, L, W) 출력하므로
    devkit `boxes_to_sensor`의 두 번째 단계만 적용하면 됨.)
    """
    box_ego.translate(-np.array(cam_cs["translation"]))
    box_ego.rotate(Quaternion(cam_cs["rotation"]).inverse)


def _proj_ego_points_to_image(points_ego: np.ndarray, cam_cs):
    """ego frame (N, 3) 점들을 카메라에 투영. (uv (N, 2), depth (N,)) 반환.
    trajectory 시각화용 — 박스가 아니므로 단순 변환만.
    """
    cam_t = np.array(cam_cs["translation"])
    cam_R = Quaternion(cam_cs["rotation"]).rotation_matrix  # ego←cam
    K = np.array(cam_cs["camera_intrinsic"])
    pts_cam = (points_ego - cam_t) @ cam_R  # ego→cam: R^T (p - t) (행벡터 표기)
    proj = view_points(pts_cam.T, K, normalize=True)  # (3, N)
    return proj[:2].T, pts_cam[:, 2]


def _draw_camera_devkit(ax, ts, sample_token, channel,
                        traj_gt_ego=None, traj_pred_ego=None,
                        trailer_traj_gt_ego=None, trailer_traj_pred_ego=None,
                        trailer_mask=0.0,
                        pred_states_ego=None, pred_logits=None,
                        config=None):
    """devkit 방식으로 한 카메라에 이미지 + GT/pred 박스 + trajectory 그림.

    GT 박스: `ts.get_sample_data(cam_sd)`로 sensor frame Box 객체 받기 →
             `box.render(ax, view=K)` (full 3D rotation 적용).
    Pred 박스: ego frame (x, y, h, L, W)에서 Box 만들고 ego→cam 변환 후 동일 render.
    Trajectory:
      - truck GT: lime / pred: red
      - trailer GT: cyan / pred: magenta (trailer_mask>0.5인 sample만)
    """
    sample = ts.get("sample", sample_token)
    cam_sd_token = sample["data"][channel]
    cam_sd = ts.get("sample_data", cam_sd_token)
    cam_cs = ts.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
    K = np.array(cam_cs["camera_intrinsic"])

    img_path = Path(ts.dataroot) / cam_sd["filename"]
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    ax.set_title(channel.replace("CAMERA_", ""), fontsize=10)
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.axis("off")
    imsize = (img.shape[1], img.shape[0])

    # GT 박스 — devkit이 sample timestamp 기준 ego_pose + cs로 sensor frame 변환
    _, gt_boxes_sensor, _ = ts.get_sample_data(
        cam_sd_token, box_vis_level=BoxVisibility.ANY,
    )
    for b in gt_boxes_sensor:
        # ego trailer는 별도 색(yellow)으로 강조 — ego의 일부
        is_ego_trailer = (b.name == "vehicle.ego_trailer")
        if not (_is_vehicle_category(b.name) or is_ego_trailer):
            continue
        if not box_in_image(b, K, imsize, vis_level=BoxVisibility.ANY):
            continue
        color = "yellow" if is_ego_trailer else "lime"
        lw = 2.0 if is_ego_trailer else 1.5
        b.render(ax, view=K, normalize=True,
                 colors=(color, color, color), linewidth=lw)

    # Pred 박스 — ego frame Box 객체로 만들어 sensor frame 변환 후 render
    if pred_states_ego is not None and pred_logits is not None:
        # height/z fallback — sensor frame Box들로부터 평균 높이 추정
        if gt_boxes_sensor:
            avg_h = float(np.mean([b.wlh[2] for b in gt_boxes_sensor]))
        else:
            avg_h = 1.7
        fallback_z = avg_h / 2.0  # 지면 위에 놓인 차량

        for j in range(len(pred_logits)):
            if pred_logits[j] < 0:  # logit < 0 → confidence < 0.5
                continue
            px = float(pred_states_ego[j, BoundingBox2DIndex.X])
            py = float(pred_states_ego[j, BoundingBox2DIndex.Y])
            ph = float(pred_states_ego[j, BoundingBox2DIndex.HEADING])
            pl = max(float(pred_states_ego[j, BoundingBox2DIndex.LENGTH]), 0.5)
            pw = max(float(pred_states_ego[j, BoundingBox2DIndex.WIDTH]), 0.5)

            # truckscenes Box: center, wlh=[w, l, h], orientation Quaternion (ego frame)
            box = Box(
                center=[px, py, fallback_z],
                size=[pw, pl, avg_h],
                orientation=Quaternion(axis=[0, 0, 1], angle=ph),
            )
            # ego → camera frame
            _ego_frame_box_to_sensor(box, cam_cs)
            if not box_in_image(box, K, imsize, vis_level=BoxVisibility.ANY):
                continue
            box.render(ax, view=K, normalize=True,
                       colors=("red", "red", "red"), linewidth=1.5)

    # Trajectory — z=0 (지면) 가정
    # 색깔 컨벤션:
    #   GT truck    : pink     동그라미
    #   Pred truck  : red      x
    #   GT trailer  : limegreen 동그라미
    #   Pred trailer: darkgreen x
    if traj_gt_ego is not None and len(traj_gt_ego) > 0:
        pts = np.column_stack([traj_gt_ego, np.zeros(len(traj_gt_ego))])
        uv, depth = _proj_ego_points_to_image(pts, cam_cs)
        valid = depth > 0.5
        if valid.any():
            ax.plot(uv[valid, 0], uv[valid, 1], color="pink",
                    linewidth=2, marker="o", markersize=4)

    if traj_pred_ego is not None and len(traj_pred_ego) > 0:
        pts = np.column_stack([traj_pred_ego, np.zeros(len(traj_pred_ego))])
        uv, depth = _proj_ego_points_to_image(pts, cam_cs)
        valid = depth > 0.5
        if valid.any():
            ax.plot(uv[valid, 0], uv[valid, 1], color="red",
                    linewidth=2, marker="x", markersize=4)

    # Trailer trajectory — mask=1.0 sample만
    if trailer_mask > 0.5:
        if trailer_traj_gt_ego is not None and len(trailer_traj_gt_ego) > 0:
            pts = np.column_stack([trailer_traj_gt_ego, np.zeros(len(trailer_traj_gt_ego))])
            uv, depth = _proj_ego_points_to_image(pts, cam_cs)
            valid = depth > 0.5
            if valid.any():
                ax.plot(uv[valid, 0], uv[valid, 1], color="limegreen",
                        linewidth=2, marker="o", markersize=4)
        if trailer_traj_pred_ego is not None and len(trailer_traj_pred_ego) > 0:
            pts = np.column_stack([trailer_traj_pred_ego, np.zeros(len(trailer_traj_pred_ego))])
            uv, depth = _proj_ego_points_to_image(pts, cam_cs)
            valid = depth > 0.5
            if valid.any():
                ax.plot(uv[valid, 0], uv[valid, 1], color="darkgreen",
                        linewidth=2, marker="x", markersize=4)


def _render_sample(sample_idx, features, targets, predictions, config, out_path,
                   gt_categories=None, ts=None, sample_token=None,
                   model_label=None):
    """한 샘플에 대해 camera 4개 + BEV 서브플롯 생성.
    카메라 박스 투영은 _draw_camera_devkit이 처리 (devkit Box.render 사용).
    BEV는 ego frame 5-tuple로 직접 그림 — 변경 없음.
    model_label이 있으면 figure suptitle로 어떤 ckpt의 prediction인지 표시.
    """
    fig = plt.figure(figsize=(24, 14))
    gs = gridspec.GridSpec(2, 4, height_ratios=[1, 2.2], figure=fig, hspace=0.15)
    ax_cams = [fig.add_subplot(gs[0, i]) for i in range(4)]
    ax_bev = fig.add_subplot(gs[1, :])

    # === 카메라 4개 — devkit 방식 ===
    if ts is not None and sample_token is not None:
        traj_gt_xy = targets["trajectory"].cpu().numpy()[:, :2]
        # trailer GT (mask=1인 sample만 의미 있음)
        trailer_traj_gt_xy = None
        trailer_mask = 0.0
        if "trailer_trajectory" in targets:
            trailer_traj_gt_xy = targets["trailer_trajectory"].cpu().numpy()[:, :2]
            trailer_mask = float(targets.get("trailer_mask",
                                             torch.tensor(0.0)).item())

        if predictions is not None:
            pred_traj_xy = predictions["trajectory"][0].cpu().numpy()[:, :2]
            pred_states = predictions["agent_states"][0].cpu().numpy()
            pred_logits = predictions["agent_labels"][0].cpu().numpy()
            trailer_traj_pred_xy = None
            if "trailer_trajectory" in predictions:
                trailer_traj_pred_xy = (
                    predictions["trailer_trajectory"][0].cpu().numpy()[:, :2]
                )
        else:
            pred_traj_xy = None
            pred_states = None
            pred_logits = None
            trailer_traj_pred_xy = None

        for ax_cam, (channel, _) in zip(ax_cams, CAMERA_CHANNELS):
            _draw_camera_devkit(
                ax_cam, ts, sample_token, channel,
                traj_gt_ego=traj_gt_xy,
                traj_pred_ego=pred_traj_xy,
                trailer_traj_gt_ego=trailer_traj_gt_xy,
                trailer_traj_pred_ego=trailer_traj_pred_xy,
                trailer_mask=trailer_mask,
                pred_states_ego=pred_states,
                pred_logits=pred_logits,
                config=config,
            )
    else:
        # ts 미제공 시 fallback: stitched feature image
        cam = features["camera_feature"].cpu().numpy()
        cam_img = np.transpose(cam, (1, 2, 0))
        ax_cams[0].imshow(cam_img)
        ax_cams[0].axis("off")
        for ax in ax_cams[1:]:
            ax.axis("off")

    # === BEV ===
    lidar = features["lidar_feature"].cpu().numpy()  # (1 또는 2, H, W)
    lidar_bg = lidar[0] if lidar.shape[0] == 1 else lidar.sum(0)

    # BEV 좌표 변환: ego frame (+x forward, +y LEFT) → plot frame (+up, +left)
    # plot_x = -ego_y, plot_y = ego_x. CCW π/2 rotation이므로 heading도 +π/2.
    extent = [config.lidar_min_y, config.lidar_max_y,
              config.lidar_min_x, config.lidar_max_x]
    ax_bev.imshow(lidar_bg[:, ::-1], extent=extent, cmap="gray_r",
                  origin="lower", alpha=0.6)

    # Ego truck (tractor) — heading=0 + plot 회전 π/2 → 세로 박스
    _draw_bbox(ax_bev, 0, 0, np.pi / 2, 6.9, 2.5, color="blue", label="ego truck")

    # Ego trailer — 현재 sample의 vehicle.ego_trailer ann을 ego frame으로 변환해서 그림
    if ts is not None and sample_token is not None:
        sample_obj = ts.get("sample", sample_token)
        lidar_channel = _get_reference_channel(sample_obj)
        sd_lidar = ts.get("sample_data", sample_obj["data"][lidar_channel])
        ego_pose_lidar = ts.get("ego_pose", sd_lidar["ego_pose_token"])
        ego_pos = np.array(ego_pose_lidar["translation"])
        ego_rot = Quaternion(ego_pose_lidar["rotation"])
        ego_yaw = _quaternion_to_yaw(ego_rot)
        for b in ts.get_boxes(sd_lidar["token"]):
            if b.name != "vehicle.ego_trailer":
                continue
            # global → tractor ego frame
            center = ego_rot.inverse.rotate(b.center - ego_pos)
            tx, ty = float(center[0]), float(center[1])
            box_yaw = _quaternion_to_yaw(b.orientation)
            heading_ego = (box_yaw - ego_yaw + np.pi) % (2 * np.pi) - np.pi
            length = float(b.wlh[1])
            width = float(b.wlh[0])
            _draw_bbox(ax_bev, -ty, tx, heading_ego + np.pi / 2,
                       length, width, color="orange", label="ego trailer")
    ax_bev.plot(0, 0, "b*", markersize=12)

    # GT truck trajectory — pink 동그라미
    traj_gt = targets["trajectory"].cpu().numpy()  # (num_poses, 3)
    ax_bev.plot(-traj_gt[:, 1], traj_gt[:, 0],
                color="pink", linestyle="-", marker="o",
                markersize=5, linewidth=2, label="GT truck traj")

    # GT trailer trajectory (mask=1.0 sample만) — limegreen 동그라미
    if "trailer_trajectory" in targets:
        trailer_mask_bev = float(targets.get("trailer_mask",
                                              torch.tensor(0.0)).item())
        if trailer_mask_bev > 0.5:
            tt = targets["trailer_trajectory"].cpu().numpy()
            ax_bev.plot(-tt[:, 1], tt[:, 0],
                        color="limegreen", linestyle="-", marker="o",
                        markersize=5, linewidth=2, label="GT trailer traj")

    # GT agent boxes
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
        text = gt_categories[j] if gt_categories and j < len(gt_categories) else f"{j}"
        ax_bev.text(-y, x, text, fontsize=9, color="green",
                    ha="center", va="center", fontweight="bold")

    # Predictions
    if predictions is not None:
        # Pred truck — red x
        pred_traj = predictions["trajectory"][0].cpu().numpy()
        ax_bev.plot(-pred_traj[:, 1], pred_traj[:, 0],
                    color="red", linestyle="--", marker="x",
                    markersize=6, linewidth=2, label="Pred truck traj")
        # Pred trailer — darkgreen x (mask 무관, 모델은 항상 출력. mask=0이면 무의미)
        if "trailer_trajectory" in predictions:
            pred_tt = predictions["trailer_trajectory"][0].cpu().numpy()
            ax_bev.plot(-pred_tt[:, 1], pred_tt[:, 0],
                        color="darkgreen", linestyle="--", marker="x",
                        markersize=6, linewidth=2, label="Pred trailer traj")

        pred_states = predictions["agent_states"][0].cpu().numpy()
        pred_logits = predictions["agent_labels"][0].cpu().numpy()
        for j in range(len(pred_logits)):
            if pred_logits[j] < 0:
                continue
            x = pred_states[j, BoundingBox2DIndex.X]
            y = pred_states[j, BoundingBox2DIndex.Y]
            h = pred_states[j, BoundingBox2DIndex.HEADING]
            length = pred_states[j, BoundingBox2DIndex.LENGTH]
            width = pred_states[j, BoundingBox2DIndex.WIDTH]
            score = float(1.0 / (1.0 + np.exp(-pred_logits[j])))
            _draw_bbox(ax_bev, -y, x, h + np.pi / 2, length, width,
                       color="red", label="Pred agent" if j == 0 else None)
            ax_bev.text(-y, x, f"{score:.2f}", fontsize=9, color="red",
                        ha="center", va="center", fontweight="bold")

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

    # 어떤 모델의 prediction인지 figure 상단에 박아둠 — 시각화 결과만 봐도 ckpt 식별 가능
    if model_label:
        fig.suptitle(f"model: {model_label}", fontsize=11, y=0.995)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)
    print(f"Loaded config: configs/{args.config}.py")

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

    model = None
    model_label = None
    if args.checkpoint:
        model = TransfuserModel(config=config).to(device)
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.eval()
        # ckpt 식별자: "<run_name> | epoch=N | step=K"
        # run_name은 ckpt 경로의 work_dirs/<run>/checkpoints/X.pt에서 추출
        ckpt_path = Path(args.checkpoint)
        run_name = ckpt_path.parent.parent.name if ckpt_path.parent.name == "checkpoints" \
            else ckpt_path.parent.name
        model_label = (f"{run_name} | epoch={ckpt.get('epoch', '?')} "
                       f"step={ckpt.get('global_step', '?')}")
        print(f"Loaded: {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    step = max(len(dataset) // args.num, 1)
    indices = [i * step for i in range(args.num)]

    for idx in indices:
        features, targets = dataset[idx]
        predictions = None
        if model is not None:
            with torch.no_grad():
                features_batch = {k: v.unsqueeze(0).to(device) for k, v in features.items()}
                predictions = model(features_batch)

        sample_token = dataset._sample_tokens[idx]
        gt_items = _get_gt_boxes_with_category(ts, sample_token, config)
        gt_categories = [it[-1] for it in gt_items]

        out_path = out_dir / f"sample_{idx:05d}.png"
        _render_sample(idx, features, targets, predictions, config, out_path,
                       gt_categories=gt_categories,
                       ts=ts, sample_token=sample_token,
                       model_label=model_label)
        print(f"  saved: {out_path}")

    print(f"\n{args.num} visualizations saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="v7_ground_plane",
                        help="configs/{name}.py stem. Default = paper baseline.")
    parser.add_argument("--dataroot", type=str, required=True)
    parser.add_argument("--version", type=str, default="v1.1-trainval")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--num", type=int, default=5)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--out", type=str, default="viz")
    args = parser.parse_args()
    main(args)
