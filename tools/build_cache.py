"""
TruckScenesDataset.__getitem__ 결과를 pickle.gz로 캐시.

학습이 매 step에서 (4 카메라 imread + 6 LiDAR 디코딩 + box/pose lookup)을 반복하는 비용을
사전 1회 처리로 옮겨 학습 epoch을 5~10배 가속한다.

저장 형식: <cache_dir>/<sample_token>.pkl.gz
파일 내용: pickle.dump((features, targets), f) — DataLoader collate에 그대로 들어가는 형태.

학습과 동시 실행 가능: 기본 num_workers=4로 12 코어 중 4 사용.
- 학습이 8 worker, 캐시가 4 worker → 시스템 부담 약간 ↑이지만 학습 영향 작음.
- 더 보수적으로 가려면 --num_workers 2 + nice 19로 실행.

사용:
    nice -n 10 python tools/build_cache.py \
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

from configs._base import TransfuserConfig
from dataset.dataset import TruckScenesDataset


class _CacheBuilder(torch.utils.data.Dataset):
    """기존 TruckScenesDataset을 감싸 cache 저장만 담당.

    DataLoader가 worker들로 __getitem__을 병렬 호출 → 각 worker가 한 sample 처리·저장.
    이미 캐시된 sample은 즉시 skip (디스크 read 한 번으로 끝, 디코딩 없음).
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
        cache_path = self._cache_dir / f"{sample_token}.pkl.gz"
        if cache_path.exists():
            # 이미 만든 sample은 skip (resume 지원)
            return ("skipped", sample_token)
        # compute (4 카메라 + 6 LiDAR + box/pose lookup)
        features, targets = self._ds[idx]
        # atomic 저장: 임시 파일에 쓰고 rename — 중간에 중단되어도 깨진 파일 안 남음
        tmp_path = cache_path.with_suffix(".pkl.gz.tmp")
        with gzip.open(tmp_path, "wb", compresslevel=self._compresslevel) as f:
            pickle.dump((features, targets), f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(cache_path)
        return ("saved", sample_token)


def _collate_passthrough(batch):
    """DataLoader가 stack 안 시도하도록 그대로 통과."""
    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, default="data/man-truckscenes")
    parser.add_argument("--version", type=str, default="v1.2-trainval")
    parser.add_argument("--cache_dir", type=str, default="data/cache/v1.2-trainval")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="학습과 동시 실행 시 4 권장 (12 코어 중 4 사용)")
    parser.add_argument("--split", type=str, default="all",
                        choices=["all", "train", "val"],
                        help="all = train + val 모두 캐시. 기본 권장.")
    parser.add_argument("--compresslevel", type=int, default=4,
                        help="gzip 압축 레벨 (1~9). 4가 속도/크기 균형.")
    parser.add_argument("--log_every", type=int, default=200)
    args = parser.parse_args()

    config = TransfuserConfig()

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
    n_saved = 0
    n_skipped = 0
    t0 = time.time()
    for i, batch in enumerate(loader):
        status, _ = batch[0]
        if status == "saved":
            n_saved += 1
        else:
            n_skipped += 1

        if (i + 1) % args.log_every == 0 or i + 1 == n:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{n}] saved={n_saved}, skipped={n_skipped}, "
                  f"{rate:.1f} samples/s, elapsed={elapsed:.0f}s, ETA={eta:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. saved={n_saved}, skipped={n_skipped}", flush=True)

    # 디스크 사용량 확인
    cache_path = Path(args.cache_dir)
    files = list(cache_path.glob("*.pkl.gz"))
    total_size = sum(f.stat().st_size for f in files)
    avg_size = total_size / max(len(files), 1)
    print(f"Cache: {len(files)} files, total {total_size/1e9:.2f} GB, "
          f"avg {avg_size/1e6:.2f} MB/sample", flush=True)


if __name__ == "__main__":
    main()
