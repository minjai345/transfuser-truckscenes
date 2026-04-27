"""
Scene을 ego_trailer 유무로 분리해서 json으로 저장.

출력 구조:
{
  "version": "v1.1-trainval",
  "total_scenes": 598,
  "with_ego_trailer":    {"count": N, "scenes": [{"name": ..., "token": ...}, ...]},
  "without_ego_trailer": {"count": M, "scenes": [...]}
}

사용:
  python tools/dump_trailer_scenes.py --version v1.1-trainval \
      --output data/scene_trailer_split.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, default="data/man-truckscenes")
    parser.add_argument("--version", type=str, default="v1.1-trainval")
    parser.add_argument("--output", type=str, default="data/scene_trailer_split.json")
    args = parser.parse_args()

    print(f"Loading TruckScenes {args.version}...", flush=True)
    from truckscenes.truckscenes import TruckScenes
    ts = TruckScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    n_scenes = len(ts.scene)
    print(f"Total scenes: {n_scenes}", flush=True)

    # category lookup — 확실히 ego_trailer 카테고리 token만 매칭
    ego_trailer_cat = next((c for c in ts.category if c["name"] == "vehicle.ego_trailer"), None)
    if ego_trailer_cat is None:
        print("ERROR: vehicle.ego_trailer 카테고리 미정의", flush=True)
        sys.exit(1)

    with_trailer = []
    without_trailer = []

    t0 = time.time()
    for s_i, scene in enumerate(ts.scene):
        sample_token = scene["first_sample_token"]
        has_et = False
        while sample_token:
            sample = ts.get("sample", sample_token)
            for ann_tok in sample["anns"]:
                ann = ts.get("sample_annotation", ann_tok)
                cat_name = ts.get("category", ann["category_token"])["name"] \
                    if "category_token" in ann else ann.get("category_name", "")
                if cat_name == "vehicle.ego_trailer":
                    has_et = True
                    break
            if has_et:
                break
            sample_token = sample.get("next", "")

        entry = {"name": scene["name"], "token": scene["token"]}
        (with_trailer if has_et else without_trailer).append(entry)

        if (s_i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{s_i+1}/{n_scenes}] elapsed={elapsed:.1f}s "
                  f"(with={len(with_trailer)}, without={len(without_trailer)})", flush=True)

    out = {
        "version": args.version,
        "total_scenes": n_scenes,
        "with_ego_trailer": {
            "count": len(with_trailer),
            "scenes": with_trailer,
        },
        "without_ego_trailer": {
            "count": len(without_trailer),
            "scenes": without_trailer,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {out_path}", flush=True)
    print(f"  with_ego_trailer:    {len(with_trailer):>4d} ({len(with_trailer)/n_scenes*100:.1f}%)", flush=True)
    print(f"  without_ego_trailer: {len(without_trailer):>4d} ({len(without_trailer)/n_scenes*100:.1f}%)", flush=True)
    print(f"\n예시 (with):    {with_trailer[:3] if with_trailer else '(none)'}", flush=True)
    print(f"예시 (without): {without_trailer[:3] if without_trailer else '(none)'}", flush=True)


if __name__ == "__main__":
    main()
