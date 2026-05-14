#!/usr/bin/env python
"""Checkpoint 평가 (val split L2 + collision rate + trailer L2).

사용:
    ./scripts/evaluate.py --ckpt <ckpt>
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
    p.add_argument("--config", default="v7_ground_plane",
                   help="configs/{name}.py — ckpt와 일치하는 버전 (default: v7_ground_plane)")
    args = p.parse_args()

    cmd = [
        sys.executable, "evaluate.py",
        "--config", args.config,
        "--dataroot", DATAROOT,
        "--version", VERSION,
        "--checkpoint", args.ckpt,
    ]
    print(f"cmd: {' '.join(cmd)}")
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
