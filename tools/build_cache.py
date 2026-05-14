"""TruckScenesDataset builder outputs를 builder-별 pickle.gz로 캐시.

학습이 매 step에서 (4 카메라 imread + 6 LiDAR 디코딩 + box/pose lookup)을 반복하는 비용을
사전 1회 처리로 옮겨 학습 epoch을 5~10배 가속한다.

저장 형식: <cache_dir>/<sample_token>/<builder_name>.pkl.gz
파일 내용: pickle.dump(builder.compute(...), f) — Builder가 반환하는 dict 그대로.

NavSim-style builder-level layout이라:
  - config가 영향 주는 builder는 unique_name에 그 정보를 인코딩 → 다른 config로 빌드된
    cache file은 다른 이름을 가져 silent mismatch 불가능 (학습 시 file not found).
  - resume은 builder 단위 — 새 builder 추가하거나 일부 builder가 빠진 sample도
    이미 만든 file은 그대로, 빠진 것만 재계산.
  - 여러 model/config가 같은 <sample_token>/ 안에 본인 file들을 공존시킬 수 있음.

학습과 동시 실행 가능: 기본 num_workers=4로 12 코어 중 4 사용.

사용:
    nice -n 10 python tools/build_cache.py \
        --config v9_cmd_no_status \
        --dataroot data/man-truckscenes --version v1.2-trainval \
        --cache_dir data/cache/v1.2-trainval --num_workers 4
"""

import argparse
import gzip
import pickle
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from configs import load_config
from configs._base import TransfuserConfig
from dataset.dataset import TruckScenesDataset


class _CacheBuilder(torch.utils.data.Dataset):
    """TruckScenesDataset의 builder별 output을 disk에 저장.

    DataLoader가 worker들로 __getitem__을 병렬 호출 → 각 worker가 한 sample 처리.
    이미 모든 builder file이 있는 sample은 즉시 skip, 일부만 빠진 sample은
    빠진 builder만 재계산 (resume 안전).
    """

    def __init__(self, base_ds: TruckScenesDataset, cache_dir: Path, compresslevel: int):
        self._ds = base_ds
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._compresslevel = compresslevel

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        sample_token = self._ds._sample_tokens[idx]
        sample_dir = self._cache_dir / sample_token
        sample_dir.mkdir(parents=True, exist_ok=True)

        # builder별 file path. 이미 있으면 skip, 없는 것만 계산.
        missing = []
        for b in self._ds._builders:
            file_path = sample_dir / f"{b.get_unique_name()}.pkl.gz"
            if not file_path.exists():
                missing.append((b, file_path))

        if not missing:
            return ("skipped", sample_token, 0)

        # 한 sample은 한 번만 ts.get("sample", ...)로 로드해서 builder들이 공유
        sample = self._ds._ts.get("sample", sample_token)
        for b, file_path in missing:
            output = b.compute(self._ds, sample)
            # atomic 저장: 임시 파일에 쓰고 rename
            tmp_path = file_path.with_suffix(".pkl.gz.tmp")
            with gzip.open(tmp_path, "wb", compresslevel=self._compresslevel) as f:
                pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_path.replace(file_path)

        return ("saved", sample_token, len(missing))


def _collate_passthrough(batch):
    """DataLoader가 stack 안 시도하도록 그대로 통과."""
    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, default="data/man-truckscenes")
    parser.add_argument("--version", type=str, default="v1.2-trainval")
    parser.add_argument("--cache_dir", type=str, default="data/cache/v1.2-trainval")
    parser.add_argument("--config", type=str, default=None,
                        help="configs/<name>.py — only fields used by builders "
                             "(LiDAR range/channels, camera size, num_poses, "
                             "hitch settings) actually affect cache filenames. "
                             "None means base TransfuserConfig() defaults.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="학습과 동시 실행 시 4 권장 (12 코어 중 4 사용)")
    parser.add_argument("--split", type=str, default="all",
                        choices=["all", "train", "val"],
                        help="all = train + val 모두 캐시. 기본 권장.")
    parser.add_argument("--compresslevel", type=int, default=4,
                        help="gzip 압축 레벨 (1~9). 4가 속도/크기 균형.")
    parser.add_argument("--log_every", type=int, default=200)
    args = parser.parse_args()

    if args.config is not None:
        config = load_config(args.config)
        print(f"Using config: configs/{args.config}.py "
              f"(use_ground_plane={config.use_ground_plane}, "
              f"lidar_range=[{config.lidar_min_x},{config.lidar_max_x}]x"
              f"[{config.lidar_min_y},{config.lidar_max_y}], "
              f"lidar_resolution={config.lidar_resolution_width}x{config.lidar_resolution_height})",
              flush=True)
    else:
        config = TransfuserConfig()
        print("Using TransfuserConfig() defaults. "
              "Pass --config <name> to build cache for a specific training config.",
              flush=True)

    print(f"Loading TruckScenes {args.version}...", flush=True)
    from truckscenes.truckscenes import TruckScenes
    from truckscenes.utils.splits import create_splits_scenes
    ts = TruckScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    # split 결정 — 기본 all (train + val 모두 캐시)
    if args.split == "all":
        splits = create_splits_scenes()
        names = set(splits.get("train", [])) | set(splits.get("val", []))
        split_tokens = [s["token"] for s in ts.scene if s["name"] in names]
    else:
        splits = create_splits_scenes()
        names = set(splits[args.split])
        split_tokens = [s["token"] for s in ts.scene if s["name"] in names]

    base_ds = TruckScenesDataset(ts=ts, config=config,
                                 num_future_samples=config.num_poses,
                                 split_tokens=split_tokens)
    print(f"Total samples to cache: {len(base_ds)}", flush=True)
    print(f"Cache dir: {args.cache_dir}", flush=True)
    print(f"Builders: {[b.get_unique_name() for b in base_ds._builders]}", flush=True)
    print(f"Workers: {args.num_workers}, gzip level: {args.compresslevel}", flush=True)

    builder = _CacheBuilder(base_ds, Path(args.cache_dir), args.compresslevel)
    loader = DataLoader(
        builder,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate_passthrough,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    n = len(builder)
    n_fully_skipped = 0   # 모든 builder file 이미 있어 sample 전체 skip
    n_saved = 0           # 한 builder 이상 새로 저장
    n_builders_written = 0
    t0 = time.time()
    for i, batch in enumerate(loader):
        status, _, builders_written = batch[0]
        if status == "saved":
            n_saved += 1
            n_builders_written += builders_written
        else:
            n_fully_skipped += 1

        if (i + 1) % args.log_every == 0 or i + 1 == n:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{n}] saved={n_saved} (builders_written={n_builders_written}), "
                  f"fully_skipped={n_fully_skipped}, "
                  f"{rate:.1f} samples/s, elapsed={elapsed:.0f}s, ETA={eta:.0f}s",
                  flush=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. saved={n_saved} samples "
          f"({n_builders_written} builder files), fully_skipped={n_fully_skipped}",
          flush=True)

    # 디스크 사용량 확인 — 모든 sample dir 안 builder file 합계
    cache_path = Path(args.cache_dir)
    files = list(cache_path.glob("*/*.pkl.gz"))
    total_size = sum(f.stat().st_size for f in files)
    avg_per_sample = total_size / max(n, 1)
    print(f"Cache: {len(files)} builder files across {n} samples, "
          f"total {total_size/1e9:.2f} GB, avg {avg_per_sample/1e6:.2f} MB/sample",
          flush=True)


if __name__ == "__main__":
    main()
