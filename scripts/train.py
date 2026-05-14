#!/usr/bin/env python
"""학습 시작 또는 resume.

사용:
    ./scripts/train.py                                  # 새 학습 trailer_v2, 20 epochs
    ./scripts/train.py --run_name my_run                # custom run name
    ./scripts/train.py --epochs 30                      # epochs 변경
    ./scripts/train.py --resume <ckpt_path>             # 이어서 학습 (run_name 자동 추정)
    ./scripts/train.py --resume <ckpt> --epochs 30      # 이어서 + total epochs 변경
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
    p.add_argument("--config", default="v7_ground_plane",
                   help="configs/{name}.py stem (default: v7_ground_plane — paper baseline).")
    p.add_argument("--run_name", default="trailer_v4",
                   help="wandb run name. resume 시 ckpt 경로에서 자동 추정.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--resume", default=None,
                   help="checkpoint path (e.g. work_dirs/<run>/checkpoints/epoch5.pt)")
    p.add_argument("--cache_dir", default=None,
                   help="Directory of pre-built sample pkl.gz cache (from tools/build_cache.py).")
    args = p.parse_args()

    # resume이면 ckpt 경로에서 run_name 자동 추정 (work_dirs/<run>/checkpoints/epoch.pt → run)
    run_name = args.run_name
    if args.resume:
        run_name = Path(args.resume).resolve().parent.parent.name
        print(f"[resume] run_name = '{run_name}' (auto from ckpt path)")

    cmd = [
        sys.executable, "-u", "train.py",
        "--config", args.config,
        "--dataroot", DATAROOT,
        "--version", VERSION,
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--lr", str(args.lr),
        "--epochs", str(args.epochs),
        "--wandb",
        "--wandb_run_name", run_name,
    ]
    if args.resume:
        cmd += ["--resume", args.resume]
    if args.cache_dir:
        cmd += ["--cache_dir", args.cache_dir]

    suffix = "_resume" if args.resume else ""
    log_path = Path(f"/tmp/train_{run_name}{suffix}.log")

    print(f"cwd:  {PROJECT_ROOT}")
    print(f"cmd:  {' '.join(cmd)}")
    print(f"log:  {log_path}")
    print()

    # tee — stdout + log file 동시 기록
    with open(log_path, "w") as f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                f.write(line)
                f.flush()
        except KeyboardInterrupt:
            proc.terminate()
            raise
        proc.wait()
        sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
