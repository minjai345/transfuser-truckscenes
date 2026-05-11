#!/usr/bin/env python
"""val split N개 sample을 PNG로 시각화. checkpoint 주면 prediction까지 오버레이.

사용:
    ./scripts/visualize.py                              # GT만 5개
    ./scripts/visualize.py --ckpt <ckpt>                # ckpt prediction까지
    ./scripts/visualize.py --ckpt <ckpt> --num 10 --out viz/my_out
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

DATAROOT = "data/man-truckscenes"
VERSION = "v1.1-trainval"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=None, help="checkpoint path (없으면 GT만)")
    p.add_argument("--num", type=int, default=5)
    p.add_argument("--out", default="viz")
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--config", default="v7_ground_plane",
                   help="configs/{name}.py (default: v7_ground_plane)")
    args = p.parse_args()

    cmd = [
        sys.executable, "tools/visualize.py",
        "--config", args.config,
        "--dataroot", DATAROOT,
        "--version", VERSION,
        "--split", args.split,
        "--num", str(args.num),
        "--out", args.out,
    ]
    if args.ckpt:
        cmd += ["--checkpoint", args.ckpt]

    print(f"cmd: {' '.join(cmd)}")
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
