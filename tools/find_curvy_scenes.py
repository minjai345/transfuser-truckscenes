"""
val split에서 회전이 큰 (curvy) scene 찾기.
ego_pose의 yaw 변화량 합산 — articulation 효과 보기 좋은 scene.

결과는 data/scene_curvy_split.json에 캐시 저장. 시각화 도구는 이 JSON을 읽어 사용.
재계산 필요할 때만 다시 돌리면 됨.
"""

import json
import sys
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))

from truckscenes.truckscenes import TruckScenes
from truckscenes.utils.splits import create_splits_scenes

from dataset.dataset import _get_reference_channel, _quaternion_to_yaw

OUT_PATH = _HERE.parent.parent / "data" / "scene_curvy_split.json"


def main():
    print("Loading...", flush=True)
    ts = TruckScenes(version="v1.2-trainval", dataroot="data/man-truckscenes",
                     verbose=False)
    splits = create_splits_scenes()
    val_names = set(splits["val"])
    val_scenes = [s for s in ts.scene if s["name"] in val_names]

    results = []
    for sc_i, scene in enumerate(val_scenes):
        sample_token = scene["first_sample_token"]
        yaws = []
        positions = []
        while sample_token:
            sample = ts.get("sample", sample_token)
            lidar_channel = _get_reference_channel(sample)
            sd = ts.get("sample_data", sample["data"][lidar_channel])
            ep = ts.get("ego_pose", sd["ego_pose_token"])
            yaws.append(_quaternion_to_yaw(Quaternion(ep["rotation"])))
            positions.append(ep["translation"][:2])
            sample_token = sample.get("next", "")

        if len(yaws) < 5:
            continue
        # cumulative yaw change (unwrap 후)
        yaws_unwrapped = np.unwrap(yaws)
        total_yaw_deg = float(np.degrees(abs(yaws_unwrapped[-1] - yaws_unwrapped[0])))
        # 또는 점진적 변화 합 (여러 번 회전 시)
        cum_yaw_deg = float(np.degrees(np.sum(np.abs(np.diff(yaws_unwrapped)))))
        # ego 이동 거리
        positions = np.array(positions)
        dist = float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))

        results.append({
            "idx": sc_i,
            "name": scene["name"],
            "token": scene["token"],
            "n_samples": len(yaws),
            "total_yaw_deg": total_yaw_deg,
            "cum_yaw_deg": cum_yaw_deg,
            "dist_m": dist,
            "curvature": cum_yaw_deg / max(dist, 1.0),  # deg/m
        })

    # 정렬: cumulative yaw 큰 순
    results.sort(key=lambda r: -r["cum_yaw_deg"])

    print(f"\nTop 15 curvy scenes (val split, n={len(results)}):", flush=True)
    print(f"{'idx':>4} {'cum_yaw':>9} {'total_yaw':>10} {'dist_m':>9} "
          f"{'curve(deg/m)':>14} {'n_samp':>7}  scene", flush=True)
    print("-" * 100, flush=True)
    for r in results[:15]:
        print(f"{r['idx']:>4} {r['cum_yaw_deg']:>8.1f}° {r['total_yaw_deg']:>9.1f}° "
              f"{r['dist_m']:>8.1f} {r['curvature']:>13.3f}  {r['n_samples']:>6} "
              f"  {r['name']}", flush=True)

    # JSON 캐시 저장 — 시각화 도구가 이걸 읽어 사용
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({"split": "val", "scenes": results}, f, indent=2)
    print(f"\nSaved to {OUT_PATH} ({len(results)} scenes, sorted by cum_yaw_deg desc)",
          flush=True)


if __name__ == "__main__":
    main()
