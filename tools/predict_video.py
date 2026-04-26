"""
Scene 단위로 prediction을 frame별 BEV+camera 시각화한 후 mp4 영상으로 합침.

각 frame:
  - 좌측: stitched 4-camera 이미지
  - 우측: BEV (LiDAR + ego + GT trajectory/agents + Pred trajectory/agents)
영상은 imageio + ffmpeg로 인코딩 (system ffmpeg 불필요).

사용법:
    python tools/predict_video.py --dataroot <path> --checkpoint <ckpt.pt> \
        --scene_idx 0 --out viz/scene0.mp4 --fps 2
"""

import argparse
import sys
import tempfile
from pathlib import Path

import imageio.v2 as imageio  # v2 API: 단순 reader/writer
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.config import TransfuserConfig
from model.model import TransfuserModel
from dataset.dataset import TruckScenesDataset

# 동일 폴더의 viz 모듈에서 렌더링 함수와 카테고리 헬퍼 재사용
from visualize import _render_sample, _get_gt_boxes_with_category


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = TransfuserConfig()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from truckscenes.truckscenes import TruckScenes
    from truckscenes.utils.splits import create_splits_scenes

    print(f"Loading TruckScenes {args.version}...")
    ts = TruckScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    # Scene 선택: split 내 scene_idx번째 scene
    splits = create_splits_scenes()
    split_names = set(splits[args.split])
    candidate_scenes = [s for s in ts.scene if s["name"] in split_names]
    if args.scene_idx >= len(candidate_scenes):
        raise ValueError(f"scene_idx {args.scene_idx} >= {len(candidate_scenes)}")
    scene = candidate_scenes[args.scene_idx]
    print(f"Scene {args.scene_idx}: {scene['name']} ({scene['nbr_samples']} samples)")

    # 이 scene만 포함하는 dataset 생성 — 마지막 num_poses 프레임은 future가 없어 제외됨
    dataset = TruckScenesDataset(
        ts=ts, config=config, num_future_samples=config.num_poses,
        split_tokens=[scene["token"]],
    )
    print(f"Renderable frames: {len(dataset)}")

    # 모델 로드
    model = None
    if args.checkpoint:
        model = TransfuserModel(config=config).to(device)
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.eval()
        print(f"Loaded: {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    # 임시 폴더에 frame PNG 저장 → mp4로 합침
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        frame_paths = []
        for idx in range(len(dataset)):
            features, targets = dataset[idx]
            predictions = None
            if model is not None:
                with torch.no_grad():
                    fb = {k: v.unsqueeze(0).to(device) for k, v in features.items()}
                    predictions = model(fb)

            # GT 박스에 카테고리 라벨링용
            sample_token = dataset._sample_tokens[idx]
            gt_items = _get_gt_boxes_with_category(ts, sample_token, config)
            gt_categories = [it[-1] for it in gt_items]

            frame_path = tmpdir / f"frame_{idx:05d}.png"
            _render_sample(idx, features, targets, predictions, config, frame_path,
                           gt_categories=gt_categories,
                           ts=ts, sample_token=sample_token)
            frame_paths.append(frame_path)
            if (idx + 1) % 5 == 0 or idx == len(dataset) - 1:
                print(f"  frame {idx+1}/{len(dataset)}")

        # mp4 인코딩
        print(f"\nEncoding video → {out_path} (fps={args.fps})...")
        with imageio.get_writer(str(out_path), fps=args.fps,
                                 codec="libx264", quality=8) as writer:
            for fp in frame_paths:
                writer.append_data(imageio.imread(str(fp)))

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, required=True)
    parser.add_argument("--version", type=str, default="v1.1-trainval")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--scene_idx", type=int, default=0,
                        help="Scene index within split (0-based)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out", type=str, default="viz/scene.mp4")
    parser.add_argument("--fps", type=int, default=2,
                        help="Sample은 0.5s 간격이라 fps=2가 실시간 속도")
    args = parser.parse_args()
    main(args)
