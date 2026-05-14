#!/usr/bin/env python
"""한 scene의 prediction을 frame별 BEV+camera 시각화 후 mp4로 합침.

사용:
    ./scripts/predict_video.py --ckpt <ckpt> --scene_idx 12
    ./scripts/predict_video.py --ckpt <ckpt> --scene_idx 12 --out viz/my.mp4 --fps 4
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

DATAROOT = "data/man-truckscenes"
VERSION = "v1.2-trainval"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="checkpoint path")
    p.add_argument("--scene_idx", type=int, required=True,
                   help="val split 내 scene index (0-based)")
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--out", default=None,
                   help="output mp4 path (default: viz/scene<idx>.mp4)")
    p.add_argument("--fps", type=int, default=2,
                   help="2가 실시간 (sample 0.5s 간격)")
    p.add_argument("--config", default="v7_ground_plane",
                   help="configs/{name}.py (default: v7_ground_plane)")
    args = p.parse_args()

    out = args.out or f"viz/scene{args.scene_idx}.mp4"
    cmd = [
        sys.executable, "tools/predict_video.py",
        "--config", args.config,
        "--dataroot", DATAROOT,
        "--version", VERSION,
        "--split", args.split,
        "--scene_idx", str(args.scene_idx),
        "--checkpoint", args.ckpt,
        "--out", out,
        "--fps", str(args.fps),
    ]
    print(f"cmd: {' '.join(cmd)}")
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
