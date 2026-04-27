"""
Evaluation script for TransFuser on MAN TruckScenes.
Metrics: L2 displacement error and collision rate at 1s, 2s, 3s.
Following the standard open-loop planning evaluation protocol (UniAD, VAD, ST-P3).
"""

import argparse

import numpy as np
import torch
from pyquaternion import Quaternion
from shapely.geometry import Polygon

from model.config import TransfuserConfig
from model.model import TransfuserModel
from dataset.dataset import (
    TruckScenesDataset,
    _is_vehicle_category,
    _get_reference_channel,
    _quaternion_to_yaw,
)

# Evaluation time horizons (seconds)
EVAL_HORIZONS = [1.0, 2.0, 3.0]


def _oriented_box_polygon(x, y, heading, length, width):
    """Create a Shapely Polygon for an oriented bounding box."""
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    hl, hw = length / 2, width / 2
    corners = np.array([
        [hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw],
    ])
    rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    corners = corners @ rot.T + np.array([x, y])
    return Polygon(corners)


def _get_future_agent_boxes(ts, sample_token, step, config):
    """Get agent boxes at a future timestep, transformed to current ego frame.

    Args:
        ts: TruckScenes devkit instance.
        sample_token: current sample token.
        step: future step index (0-based, so step=0 means the next sample).
        config: TransfuserConfig.

    Returns:
        List of (x, y, heading, length, width) tuples in current ego frame.
    """
    sample = ts.get("sample", sample_token)
    lidar_channel = _get_reference_channel(sample)

    # Current ego pose (reference frame)
    sd = ts.get("sample_data", sample["data"][lidar_channel])
    current_ego = ts.get("ego_pose", sd["ego_pose_token"])
    current_pos = np.array(current_ego["translation"][:2])
    current_yaw = _quaternion_to_yaw(Quaternion(current_ego["rotation"]))

    # Walk to future sample
    future_token = sample_token
    for _ in range(step + 1):
        future_sample = ts.get("sample", future_token)
        next_token = future_sample.get("next", "")
        if not next_token:
            return []
        future_token = next_token

    # Get boxes at future timestep
    future_sample = ts.get("sample", future_token)
    if lidar_channel not in future_sample["data"]:
        return []
    future_sd = ts.get("sample_data", future_sample["data"][lidar_channel])
    boxes = ts.get_boxes(future_sd["token"])

    agent_boxes = []
    for box in boxes:
        if not _is_vehicle_category(box.name):
            continue

        # box.center is in global frame
        center_global = box.center[:2]
        delta = center_global - current_pos
        cos_yaw, sin_yaw = np.cos(-current_yaw), np.sin(-current_yaw)
        local_x = delta[0] * cos_yaw - delta[1] * sin_yaw
        local_y = delta[0] * sin_yaw + delta[1] * cos_yaw

        # Skip boxes far outside perception range
        if abs(local_x) > config.lidar_max_x * 2 or abs(local_y) > config.lidar_max_y * 2:
            continue

        box_yaw = _quaternion_to_yaw(box.orientation)
        heading = (box_yaw - current_yaw + np.pi) % (2 * np.pi) - np.pi
        length, width = box.wlh[1], box.wlh[0]
        agent_boxes.append((local_x, local_y, heading, length, width))

    return agent_boxes


def _check_collision(ego_pose, ego_length, ego_width, agent_boxes):
    """Check if ego bbox at given pose collides with any agent bbox."""
    if len(agent_boxes) == 0:
        return False

    ego_poly = _oriented_box_polygon(
        ego_pose[0], ego_pose[1], ego_pose[2], ego_length, ego_width,
    )

    for box in agent_boxes:
        agent_poly = _oriented_box_polygon(*box)
        if ego_poly.intersects(agent_poly):
            return True

    return False


def run_evaluation(
    model,
    ts,
    dataset,
    config,
    device,
    ego_length: float = 6.9,
    ego_width: float = 2.5,
    log_interval: int = 50,
    verbose: bool = True,
):
    """Run L2 + collision evaluation on a dataset. Returns metrics dict.

    학습 루프에서도 호출 가능하도록 모델/데이터셋을 인자로 받는 형태.
    반환값: {"l2/1s": x, "l2/2s": x, "l2/3s": x, "l2/avg": x,
             "col/1s": x, "col/2s": x, "col/3s": x, "col/avg": x}
    """
    # Pose indices for each evaluation horizon (0.5s 간격이면 1s→idx 1, 2s→idx 3, 3s→idx 5)
    horizon_indices = {
        h: int(h / config.trajectory_sampling_interval) - 1 for h in EVAL_HORIZONS
    }

    # Metrics accumulators
    l2_errors = {h: [] for h in EVAL_HORIZONS}
    collisions = {h: [] for h in EVAL_HORIZONS}
    # trailer L2: trailer_mask=1인 sample에서만 집계 (with-trailer scene만 포함)
    trailer_l2_errors = {h: [] for h in EVAL_HORIZONS}

    num_samples = len(dataset)
    if verbose:
        print(f"Evaluating {num_samples} samples...")

    # 기존 모드를 저장해서 끝나면 복원 (학습 루프에서 eval→train 전환 시 중요)
    prev_mode = model.training
    model.eval()
    try:
        with torch.no_grad():
            for i in range(num_samples):
                features, targets = dataset[i]
                sample_token = dataset._sample_tokens[i]

                features_batch = {k: v.unsqueeze(0).to(device) for k, v in features.items()}
                predictions = model(features_batch)

                pred_traj = predictions["trajectory"][0].cpu().numpy()  # (8, 3)
                gt_traj = targets["trajectory"].numpy()  # (8, 3)

                # trailer prediction (선택적). target에 mask=1일 때만 metric에 포함.
                trailer_mask = float(targets.get("trailer_mask", torch.tensor(0.0)).item())
                pred_trailer = None
                gt_trailer = None
                if (
                    "trailer_trajectory" in predictions
                    and "trailer_trajectory" in targets
                    and trailer_mask > 0.5
                ):
                    pred_trailer = predictions["trailer_trajectory"][0].cpu().numpy()
                    gt_trailer = targets["trailer_trajectory"].numpy()

                for h in EVAL_HORIZONS:
                    idx = horizon_indices[h]

                    l2 = np.linalg.norm(pred_traj[idx, :2] - gt_traj[idx, :2])
                    l2_errors[h].append(l2)

                    agent_boxes = _get_future_agent_boxes(ts, sample_token, idx, config)
                    collision = _check_collision(
                        pred_traj[idx], ego_length, ego_width, agent_boxes,
                    )
                    collisions[h].append(float(collision))

                    # trailer L2 (mask=1 sample만)
                    if pred_trailer is not None:
                        tl2 = np.linalg.norm(pred_trailer[idx, :2] - gt_trailer[idx, :2])
                        trailer_l2_errors[h].append(tl2)

                if verbose and (i + 1) % log_interval == 0:
                    print(f"  [{i + 1}/{num_samples}]")
    finally:
        model.train(prev_mode)

    # 결과 집계
    metrics = {}
    l2_means = []
    col_means = []
    trailer_means = []
    for h in EVAL_HORIZONS:
        l2_m = float(np.mean(l2_errors[h])) if l2_errors[h] else float("nan")
        col_m = float(np.mean(collisions[h])) * 100 if collisions[h] else float("nan")
        metrics[f"l2/{int(h)}s"] = l2_m
        metrics[f"col/{int(h)}s"] = col_m
        l2_means.append(l2_m)
        col_means.append(col_m)

        # trailer L2 — sample 수가 충분할 때만 의미 있음
        if trailer_l2_errors[h]:
            t_m = float(np.mean(trailer_l2_errors[h]))
            metrics[f"trailer_l2/{int(h)}s"] = t_m
            trailer_means.append(t_m)
        else:
            metrics[f"trailer_l2/{int(h)}s"] = float("nan")

    metrics["l2/avg"] = float(np.mean(l2_means))
    metrics["col/avg"] = float(np.mean(col_means))
    if trailer_means:
        metrics["trailer_l2/avg"] = float(np.mean(trailer_means))
        metrics["trailer_l2/count"] = float(len(trailer_l2_errors[EVAL_HORIZONS[0]]))
    else:
        metrics["trailer_l2/avg"] = float("nan")
        metrics["trailer_l2/count"] = 0.0

    if verbose:
        _print_results(metrics)

    return metrics


def _print_results(metrics):
    """Pretty-print evaluation metrics."""
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    print("\nTruck L2 (m) ", end="")
    for h in EVAL_HORIZONS:
        print(f"| {h:.0f}s     ", end="")
    print("| Avg")
    print("-" * 60)
    print("             ", end="")
    for h in EVAL_HORIZONS:
        print(f"| {metrics[f'l2/{int(h)}s']:<7.2f}", end="")
    print(f"| {metrics['l2/avg']:.2f}")

    print("\nCol (%)      ", end="")
    for h in EVAL_HORIZONS:
        print(f"| {h:.0f}s     ", end="")
    print("| Avg")
    print("-" * 60)
    print("             ", end="")
    for h in EVAL_HORIZONS:
        print(f"| {metrics[f'col/{int(h)}s']:<7.2f}", end="")
    print(f"| {metrics['col/avg']:.2f}")

    # Trailer L2 (with-trailer sample만 평균; nan이면 trailer 없는 set)
    n_trailer = int(metrics.get("trailer_l2/count", 0))
    if n_trailer > 0:
        print(f"\nTrailer L2 (m) [n={n_trailer}] ", end="")
        for h in EVAL_HORIZONS:
            print(f"| {h:.0f}s     ", end="")
        print("| Avg")
        print("-" * 60)
        print("             ", end="")
        for h in EVAL_HORIZONS:
            v = metrics.get(f"trailer_l2/{int(h)}s", float("nan"))
            print(f"| {v:<7.2f}", end="")
        print(f"| {metrics['trailer_l2/avg']:.2f}")
    print()


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = TransfuserConfig()

    from truckscenes.truckscenes import TruckScenes
    print(f"Loading TruckScenes {args.version} from {args.dataroot}...")
    ts = TruckScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    # 평가는 공식 val split 사용
    from truckscenes.utils.splits import create_splits_scenes
    val_scene_names = set(create_splits_scenes()["val"])
    val_scene_tokens = [s["token"] for s in ts.scene if s["name"] in val_scene_names]
    print(f"Val split: {len(val_scene_tokens)} scenes")

    dataset = TruckScenesDataset(
        ts=ts, config=config, num_future_samples=config.num_poses,
        split_tokens=val_scene_tokens,
    )

    # Load model
    model = TransfuserModel(config=config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    print(
        f"Loaded checkpoint: {args.checkpoint} "
        f"(epoch {checkpoint.get('epoch', '?')})"
    )

    run_evaluation(
        model=model,
        ts=ts,
        dataset=dataset,
        config=config,
        device=device,
        ego_length=args.ego_length,
        ego_width=args.ego_width,
        log_interval=args.log_interval,
        verbose=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TransFuser on TruckScenes")
    parser.add_argument("--dataroot", type=str, required=True)
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--ego_length", type=float, default=6.9,
                        help="Ego vehicle length in meters (MAN truck default)")
    parser.add_argument("--ego_width", type=float, default=2.5,
                        help="Ego vehicle width in meters (MAN truck default)")
    parser.add_argument("--log_interval", type=int, default=50)

    args = parser.parse_args()
    evaluate(args)
