#!/home/minjai/miniforge3/envs/truckscenes/bin/python
"""
data/scene_curvy_split.json (find_curvy_scenes.py로 생성됨)에서
상위 N개 curvy scene을 골라 predict_video.py로 mp4 생성.

사용:
    ./scripts/visualize_curvy.py --ckpt <ckpt.pt>            # top 3 + fps 2 + viz/curvy/
    ./scripts/visualize_curvy.py --ckpt <ckpt.pt> --top 5
    ./scripts/visualize_curvy.py --ckpt <ckpt.pt> --top 1 --out_dir viz/test/

JSON이 없으면 친절히 안내해줌. 매번 find_curvy_scenes를 다시 돌리지 않음.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURVY_JSON = PROJECT_ROOT / "data" / "scene_curvy_split.json"
PREDICT_VIDEO = PROJECT_ROOT / "tools" / "predict_video.py"
PYTHON = "/home/minjai/miniforge3/envs/truckscenes/bin/python"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="checkpoint path")
    p.add_argument("--config", default="v7_ground_plane",
                   help="configs/{name}.py stem matching the ckpt. Default = paper baseline.")
    p.add_argument("--top", type=int, default=3, help="상위 N개 curvy scene")
    p.add_argument("--out_dir", default="viz/curvy", help="출력 폴더")
    p.add_argument("--fps", type=int, default=2)
    p.add_argument("--dataroot", default="data/man-truckscenes")
    p.add_argument("--version", default="v1.1-trainval")
    args = p.parse_args()

    if not CURVY_JSON.exists():
        print(f"[!] {CURVY_JSON} 없음. 먼저 한 번 돌려:")
        print(f"    {PYTHON} tools/find_curvy_scenes.py")
        sys.exit(1)

    with open(CURVY_JSON) as f:
        data = json.load(f)
    scenes = data["scenes"][: args.top]

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ckpt에서 run_name + epoch 뽑아 mp4 파일명 prefix로 사용 — 어떤 모델 결과인지 명확
    ckpt_path = Path(args.ckpt)
    run_name = ckpt_path.parent.parent.name if ckpt_path.parent.name == "checkpoints" \
        else ckpt_path.parent.name
    epoch_tag = ckpt_path.stem  # "epoch12"
    model_tag = f"{run_name}_{epoch_tag}"

    print(f"Model: {model_tag}")
    print(f"Top {len(scenes)} curvy scenes (val):")
    for r in scenes:
        print(f"  idx={r['idx']:>3} cum_yaw={r['cum_yaw_deg']:>6.1f}°  {r['name']}")
    print()

    for r in scenes:
        out_mp4 = out_dir / f"{model_tag}__curvy_idx{r['idx']:03d}_{r['name']}.mp4"
        print(f"[render] idx={r['idx']} {r['name']} → {out_mp4}", flush=True)
        cmd = [
            PYTHON, str(PREDICT_VIDEO),
            "--config", args.config,
            "--dataroot", args.dataroot,
            "--version", args.version,
            "--split", "val",
            "--scene_idx", str(r["idx"]),
            "--checkpoint", args.ckpt,
            "--out", str(out_mp4),
            "--fps", str(args.fps),
        ]
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))

    print(f"\nSaved {len(scenes)} videos to {out_dir}/")


if __name__ == "__main__":
    main()
