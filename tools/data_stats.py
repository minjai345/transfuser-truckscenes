"""
Dataset stats — 학습 데이터가 이상한 값(전부 0, NaN 등)이 들어가고 있지 않은지 검증.
Val split 전체를 순회하면서 target/feature 분포를 요약.

사용법:
    python tools/data_stats.py --dataroot <path> --version v1.1-trainval --split val --max_samples 500
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# 프로젝트 루트를 path에 추가 (tools/에서 실행 시)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.config import TransfuserConfig
from dataset.dataset import TruckScenesDataset


def _summarize(name, arr):
    """텐서/배열을 받아 mean/std/min/max + NaN/Inf 여부 출력."""
    arr = np.asarray(arr)
    nan_cnt = int(np.isnan(arr).sum())
    inf_cnt = int(np.isinf(arr).sum())
    if arr.size == 0:
        print(f"  {name}: empty")
        return
    print(
        f"  {name}: "
        f"mean={arr.mean():.4f}, std={arr.std():.4f}, "
        f"min={arr.min():.4f}, max={arr.max():.4f}, "
        f"NaN={nan_cnt}, Inf={inf_cnt}"
    )


def main(args):
    config = TransfuserConfig()

    from truckscenes.truckscenes import TruckScenes
    from truckscenes.utils.splits import create_splits_scenes

    print(f"Loading TruckScenes {args.version}...")
    ts = TruckScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    splits = create_splits_scenes()
    split_scene_names = set(splits[args.split])
    split_tokens = [s["token"] for s in ts.scene if s["name"] in split_scene_names]
    print(f"Split '{args.split}': {len(split_tokens)} scenes")

    dataset = TruckScenesDataset(
        ts=ts, config=config, num_future_samples=config.num_poses,
        split_tokens=split_tokens,
    )

    n_total = len(dataset)
    n_check = min(n_total, args.max_samples)
    print(f"\nChecking {n_check}/{n_total} samples...")

    # accumulators
    agent_counts = []                                # sample당 valid agent 수
    traj_x = [[] for _ in range(config.num_poses)]   # timestep별 x
    traj_y = [[] for _ in range(config.num_poses)]
    traj_h = [[] for _ in range(config.num_poses)]
    status_vals = []                                 # (N, 4): vx, vy, ax, ay
    cam_mean, cam_std = [], []
    lidar_mean, lidar_max = [], []
    nan_samples = 0

    # 동일 간격으로 샘플링
    step = max(n_total // n_check, 1)
    indices = list(range(0, n_total, step))[:n_check]

    for progress, i in enumerate(indices):
        features, targets = dataset[i]

        # NaN check (feature + target 모두)
        has_nan = any(torch.isnan(v).any().item() or torch.isinf(v).any().item()
                      for v in list(features.values()) + list(targets.values()))
        if has_nan:
            nan_samples += 1

        # agent 수
        agent_counts.append(int(targets["agent_labels"].sum().item()))

        # trajectory (num_poses, 3)
        traj = targets["trajectory"].numpy()
        for t in range(config.num_poses):
            traj_x[t].append(traj[t, 0])
            traj_y[t].append(traj[t, 1])
            traj_h[t].append(traj[t, 2])

        status_vals.append(targets.get("agent_states").numpy()[:, :2])  # 사용 안함
        status_vals.append(features["status_feature"].numpy())

        cam = features["camera_feature"].numpy()
        cam_mean.append(cam.mean())
        cam_std.append(cam.std())

        lidar = features["lidar_feature"].numpy()
        lidar_mean.append(lidar.mean())
        lidar_max.append(lidar.max())

        if (progress + 1) % max(n_check // 10, 1) == 0:
            print(f"  [{progress+1}/{n_check}]")

    # === 결과 출력 ===
    print("\n" + "=" * 60)
    print("Dataset Stats Summary")
    print("=" * 60)

    print(f"\nSamples checked: {n_check}")
    print(f"NaN/Inf samples: {nan_samples}")

    # Agent count 분포
    agent_counts = np.array(agent_counts)
    print(f"\nAgents per sample: "
          f"mean={agent_counts.mean():.2f}, "
          f"min={agent_counts.min()}, max={agent_counts.max()}")
    # 히스토그램
    counter = Counter(agent_counts.tolist())
    for k in sorted(counter.keys())[:15]:
        bar = "#" * min(counter[k] // max(n_check // 40, 1), 40)
        print(f"  {k:2d} agents: {counter[k]:5d}  {bar}")
    if agent_counts.max() > 0 and (agent_counts == 0).sum() > n_check * 0.5:
        print("  *** WARNING: 50%+ 샘플에 agent가 0개. 필터 문제 가능성 ***")

    # Trajectory stats (timestep별 x, y, heading)
    print("\nTrajectory (ego-centric, per future step):")
    print(f"  step | t[s] | x mean/std/max | y mean/std/max | h mean/std/max")
    for t in range(config.num_poses):
        tsec = (t + 1) * config.trajectory_sampling_interval
        xa, ya, ha = np.array(traj_x[t]), np.array(traj_y[t]), np.array(traj_h[t])
        print(
            f"  {t:4d} | {tsec:4.1f} | "
            f"{xa.mean():+6.2f}/{xa.std():5.2f}/{xa.max():+6.2f} | "
            f"{ya.mean():+6.2f}/{ya.std():5.2f}/{ya.max():+6.2f} | "
            f"{ha.mean():+6.3f}/{ha.std():5.3f}/{ha.max():+6.3f}"
        )

    # Status feature — 마지막에 추가된 것만 사용 (zip으로 섞임 → 재구성)
    status_arr = np.array([s for s in status_vals if s.shape == (4,)])
    if len(status_arr) > 0:
        print("\nStatus feature [vx, vy, ax, ay]:")
        _summarize("vx", status_arr[:, 0])
        _summarize("vy", status_arr[:, 1])
        _summarize("ax", status_arr[:, 2])
        _summarize("ay", status_arr[:, 3])

    # Feature stats
    print("\nCamera feature (after ToTensor, expected 0~1):")
    _summarize("mean", cam_mean)
    _summarize("std", cam_std)

    print("\nLiDAR histogram feature (expected 0~1 after normalization):")
    _summarize("mean", lidar_mean)
    _summarize("max", lidar_max)

    # 주요 sanity check 플래그
    print("\n--- Sanity flags ---")
    flags = []
    if nan_samples > 0:
        flags.append(f"NaN/Inf in {nan_samples} samples")
    if agent_counts.mean() < 0.5:
        flags.append(f"Very few agents (avg {agent_counts.mean():.2f})")
    # 이동 거리: 마지막 스텝 x값이 거의 0이면 차가 거의 안 움직임 (빈번하면 수상)
    last_x = np.array(traj_x[-1])
    stationary_frac = (np.abs(last_x) < 0.5).mean()
    if stationary_frac > 0.3:
        flags.append(f"{stationary_frac*100:.1f}% samples nearly stationary at 4s")
    if np.array(cam_mean).mean() < 0.01 or np.array(cam_mean).mean() > 0.99:
        flags.append("Camera mean out of normal range")
    if np.array(lidar_mean).mean() == 0:
        flags.append("LiDAR histogram is all zero")

    if flags:
        print("WARNINGS:")
        for f in flags:
            print(f"  - {f}")
    else:
        print("모든 sanity check 통과")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, required=True)
    parser.add_argument("--version", type=str, default="v1.1-trainval")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test", "mini_train", "mini_val"])
    parser.add_argument("--max_samples", type=int, default=500, help="최대 확인 샘플 수 (균등 간격 서브샘플)")
    args = parser.parse_args()
    main(args)
