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
    VEHICLE_CATEGORIES,
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
        if box.name.split(".")[0] not in VEHICLE_CATEGORIES:
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


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = TransfuserConfig()

    # Pose indices for each evaluation horizon
    # Poses at 0.5s intervals: index 0=0.5s, 1=1.0s, 2=1.5s, 3=2.0s, 4=2.5s, 5=3.0s, ...
    horizon_indices = {
        h: int(h / config.trajectory_sampling_interval) - 1 for h in EVAL_HORIZONS
    }

    # Load TruckScenes
    from truckscenes.truckscenes import TruckScenes

    print(f"Loading TruckScenes {args.version} from {args.dataroot}...")
    ts = TruckScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    # Dataset
    dataset = TruckScenesDataset(
        ts=ts, config=config, num_future_samples=config.num_poses,
    )

    # Load model
    model = TransfuserModel(config=config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    print(
        f"Loaded checkpoint: {args.checkpoint} "
        f"(epoch {checkpoint.get('epoch', '?')})"
    )

    # Metrics accumulators
    l2_errors = {h: [] for h in EVAL_HORIZONS}
    collisions = {h: [] for h in EVAL_HORIZONS}

    num_samples = len(dataset)
    print(f"Evaluating {num_samples} samples...")

    with torch.no_grad():
        for i in range(num_samples):
            features, targets = dataset[i]
            sample_token = dataset._sample_tokens[i]

            # Single-sample forward pass
            features_batch = {k: v.unsqueeze(0).to(device) for k, v in features.items()}
            predictions = model(features_batch)

            pred_traj = predictions["trajectory"][0].cpu().numpy()  # (8, 3)
            gt_traj = targets["trajectory"].numpy()  # (8, 3)

            for h in EVAL_HORIZONS:
                idx = horizon_indices[h]

                # L2 displacement error (x, y only)
                l2 = np.linalg.norm(pred_traj[idx, :2] - gt_traj[idx, :2])
                l2_errors[h].append(l2)

                # Collision check
                agent_boxes = _get_future_agent_boxes(ts, sample_token, idx, config)
                collision = _check_collision(
                    pred_traj[idx], args.ego_length, args.ego_width, agent_boxes,
                )
                collisions[h].append(float(collision))

            if (i + 1) % args.log_interval == 0:
                print(f"  [{i + 1}/{num_samples}]")

    # === Print results ===
    print("\n" + "=" * 55)
    print("Evaluation Results")
    print("=" * 55)

    # L2 table
    avg_l2 = []
    print("\nL2 (m) ", end="")
    for h in EVAL_HORIZONS:
        print(f"| {h:.0f}s     ", end="")
    print("| Avg")
    print("-" * 55)
    print("       ", end="")
    for h in EVAL_HORIZONS:
        mean = np.mean(l2_errors[h])
        avg_l2.append(mean)
        print(f"| {mean:<7.2f}", end="")
    print(f"| {np.mean(avg_l2):.2f}")

    # Collision table
    avg_col = []
    print("\nCol (%) ", end="")
    for h in EVAL_HORIZONS:
        print(f"| {h:.0f}s     ", end="")
    print("| Avg")
    print("-" * 55)
    print("       ", end="")
    for h in EVAL_HORIZONS:
        mean = np.mean(collisions[h]) * 100
        avg_col.append(mean)
        print(f"| {mean:<7.2f}", end="")
    print(f"| {np.mean(avg_col):.2f}")

    print()


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
