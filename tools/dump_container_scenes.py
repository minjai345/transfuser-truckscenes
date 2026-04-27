"""
ego_trailer 박스 height(wlh[2])로 컨테이너 적재 여부 분류.

vehicle.ego_trailer가 있는 scene을 대상으로:
  1. 각 sample의 ego_trailer annotation height 수집
  2. scene별 median height 계산
  3. 전체 분포 출력 (percentile + ASCII histogram)
  4. --threshold 이상 → with_container, 미만 → without_container
  5. JSON 저장: data/scene_container_split.json

threshold 정하는 법:
  - default 2.5m: chassis-only(빈 trailer) ≈ 1~1.5m, 컨테이너 적재 ≈ 3m+ 사이
  - 분포 보고 --threshold X로 재실행 가능

사용:
    python tools/dump_container_scenes.py
    python tools/dump_container_scenes.py --threshold 2.0
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, default="data/man-truckscenes")
    parser.add_argument("--version", type=str, default="v1.1-trainval")
    parser.add_argument("--output", type=str,
                        default="data/scene_container_split.json")
    parser.add_argument("--threshold", type=float, default=2.5,
                        help="이 값 이상이면 컨테이너 적재로 분류 (m, default 2.5)")
    parser.add_argument("--bins", type=int, default=20)
    args = parser.parse_args()

    print(f"Loading TruckScenes {args.version}...", flush=True)
    from truckscenes.truckscenes import TruckScenes
    ts = TruckScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    # scene_token → ego_trailer annotation의 height 리스트 (sample별 1개씩 normally)
    scene_heights = defaultdict(list)
    scene_meta = {s["token"]: {"name": s["name"], "token": s["token"]}
                  for s in ts.scene}

    n_scenes = len(ts.scene)
    print(f"Total scenes: {n_scenes}", flush=True)

    t0 = time.time()
    for s_i, scene in enumerate(ts.scene):
        sample_token = scene["first_sample_token"]
        while sample_token:
            sample = ts.get("sample", sample_token)
            for ann_tok in sample["anns"]:
                ann = ts.get("sample_annotation", ann_tok)
                # category 매칭 — token 우선, fallback으로 category_name
                cat_name = (
                    ts.get("category", ann["category_token"])["name"]
                    if "category_token" in ann else ann.get("category_name", "")
                )
                if cat_name == "vehicle.ego_trailer":
                    # ann["size"] = [width, length, height]
                    h = float(ann["size"][2])
                    scene_heights[scene["token"]].append(h)
            sample_token = sample.get("next", "")

        if (s_i + 1) % 100 == 0:
            print(f"  [{s_i+1}/{n_scenes}] elapsed={time.time()-t0:.1f}s",
                  flush=True)

    # ego_trailer 있는 scene만 추림
    with_et = [(tok, hs) for tok, hs in scene_heights.items() if hs]
    print(f"\nwith_ego_trailer scenes: {len(with_et)}", flush=True)
    print(f"without_ego_trailer scenes: {n_scenes - len(with_et)} (분류 대상 아님)",
          flush=True)

    if not with_et:
        print("ERROR: ego_trailer ann 발견 안됨", flush=True)
        sys.exit(1)

    # scene별 median (한 scene 안에서 같은 trailer instance라면 height 거의 동일)
    scene_med = {tok: float(np.median(hs)) for tok, hs in with_et}
    arr = np.array(sorted(scene_med.values()))

    # 분포 통계
    print(f"\nHeight distribution (per-scene median, n={len(arr)}):", flush=True)
    print(f"  min    = {arr.min():.3f}m", flush=True)
    print(f"  p10    = {np.percentile(arr, 10):.3f}m", flush=True)
    print(f"  p25    = {np.percentile(arr, 25):.3f}m", flush=True)
    print(f"  median = {np.median(arr):.3f}m", flush=True)
    print(f"  p75    = {np.percentile(arr, 75):.3f}m", flush=True)
    print(f"  p90    = {np.percentile(arr, 90):.3f}m", flush=True)
    print(f"  max    = {arr.max():.3f}m", flush=True)
    print(f"  mean   = {arr.mean():.3f}m, std = {arr.std():.3f}m", flush=True)

    # ASCII histogram — bimodality 확인용
    counts, edges = np.histogram(arr, bins=args.bins)
    max_c = max(counts.max(), 1)
    print(f"\nHistogram ({args.bins} bins):", flush=True)
    for i in range(args.bins):
        bar = "█" * int(40 * counts[i] / max_c)
        print(f"  {edges[i]:5.2f}~{edges[i+1]:5.2f}m | {counts[i]:>4d} | {bar}",
              flush=True)

    # scene 내부 height 변동 — 같은 trailer instance면 거의 0 일 것
    var_alarm = []
    for tok, hs in with_et:
        spread = max(hs) - min(hs)
        if spread > 0.1:
            var_alarm.append((scene_meta[tok]["name"], min(hs), max(hs),
                              len(hs), spread))
    if var_alarm:
        var_alarm.sort(key=lambda x: -x[4])
        print(f"\n경고: {len(var_alarm)} scenes에서 sample 간 height 변동 > 0.1m "
              "(instance가 sample마다 다를 수 있음):", flush=True)
        for nm, lo, hi, n, sp in var_alarm[:5]:
            print(f"  {nm[:50]:<50} range=[{lo:.2f}, {hi:.2f}] "
                  f"spread={sp:.2f} (n_samples={n})", flush=True)
        if len(var_alarm) > 5:
            print(f"  ... ({len(var_alarm)-5} more)", flush=True)

    # 분류
    threshold = args.threshold
    print(f"\nThreshold: {threshold}m", flush=True)
    with_c = []
    without_c = []
    for tok, med in scene_med.items():
        entry = {**scene_meta[tok], "median_height": round(med, 3)}
        if med >= threshold:
            with_c.append(entry)
        else:
            without_c.append(entry)

    # 보기 좋게 height 순으로 정렬
    with_c.sort(key=lambda e: e["median_height"])
    without_c.sort(key=lambda e: e["median_height"])

    print(f"  with_container:    {len(with_c):>4d} "
          f"({len(with_c)/len(with_et)*100:.1f}% of with_ego_trailer)", flush=True)
    print(f"  without_container: {len(without_c):>4d} "
          f"({len(without_c)/len(with_et)*100:.1f}%)", flush=True)

    # threshold 양쪽에 가까운 sample 몇 개 출력 — sanity 검증용
    print(f"\nThreshold 인근 5개 (분류 경계):", flush=True)
    near = sorted(scene_med.items(), key=lambda kv: abs(kv[1] - threshold))[:10]
    for tok, med in near:
        side = "WITH" if med >= threshold else "WITHOUT"
        print(f"  [{side:>7}] h={med:.3f}m  {scene_meta[tok]['name']}", flush=True)

    out = {
        "version": args.version,
        "threshold_m": threshold,
        "total_with_ego_trailer": len(with_et),
        "with_container": {
            "count": len(with_c),
            "scenes": with_c,
        },
        "without_container": {
            "count": len(without_c),
            "scenes": without_c,
        },
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
