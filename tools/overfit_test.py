"""
Overfit test — 모델/loss/optimizer 경로가 실제로 학습 가능한지 검증.
샘플 N개를 메모리에 올려놓고 계속 반복 학습 → loss가 0 근처로 떨어져야 정상.
떨어지지 않으면 모델 아키텍처/loss 구현/optimizer 설정 중 하나에 버그.

사용법:
    python tools/overfit_test.py --dataroot <path> --num_samples 4 --iters 500
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs import TransfuserConfig, load_config
from model.model import TransfuserModel
from model.loss import transfuser_loss, _agent_loss
from dataset.dataset import TruckScenesDataset
import torch.nn.functional as F


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = load_config(args.config)
    print(f"Loaded config: configs/{args.config}.py")

    # Load dataset (val split: 작아서 로드 빠름)
    from truckscenes.truckscenes import TruckScenes
    from truckscenes.utils.splits import create_splits_scenes

    print(f"Loading TruckScenes {args.version}...")
    ts = TruckScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    splits = create_splits_scenes()
    val_scene_names = set(splits["val"])
    val_tokens = [s["token"] for s in ts.scene if s["name"] in val_scene_names]
    dataset = TruckScenesDataset(
        ts=ts, config=config, num_future_samples=config.num_poses,
        split_tokens=val_tokens,
    )
    print(f"Dataset: {len(dataset)} samples")

    # 처음 N개 샘플을 GPU에 미리 올려놓기
    print(f"Preloading {args.num_samples} samples...")
    features_list, targets_list = [], []
    for i in range(args.num_samples):
        f, t = dataset[i]
        features_list.append({k: v.to(device) for k, v in f.items()})
        targets_list.append({k: v.to(device) for k, v in t.items()})

    # batch로 stack
    features = {
        k: torch.stack([f[k] for f in features_list])
        for k in features_list[0]
    }
    targets = {
        k: torch.stack([t[k] for t in targets_list])
        for k in targets_list[0]
    }

    print("Feature shapes:")
    for k, v in features.items():
        print(f"  {k}: {tuple(v.shape)}")
    print("Target shapes:")
    for k, v in targets.items():
        print(f"  {k}: {tuple(v.shape)}")

    # Model + optimizer
    model = TransfuserModel(config=config).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {num_params:,}")

    # === Overfit 루프 ===
    print(f"\nOverfitting {args.num_samples} samples for {args.iters} iters...")
    losses = []
    traj_losses = []
    start = time.time()

    for it in range(args.iters):
        predictions = model(features)

        # component별 loss 분해 — 어느 head가 수렴 안 하는지 진단용
        traj_l = F.l1_loss(predictions["trajectory"], targets["trajectory"])
        agent_cls_l, agent_box_l = _agent_loss(targets, predictions, config)
        loss = (
            config.trajectory_weight * traj_l
            + config.agent_class_weight * agent_cls_l
            + config.agent_box_weight * agent_box_l
        )

        traj_l1_raw = (predictions["trajectory"] - targets["trajectory"]).abs().mean().item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(loss.item())
        traj_losses.append(traj_l1_raw)

        if it == 0 or (it + 1) % max(args.iters // 20, 1) == 0:
            elapsed = time.time() - start
            print(
                f"  iter {it+1:4d}/{args.iters} | loss {loss.item():8.4f} | "
                f"traj {traj_l.item():.4f} | cls {agent_cls_l.item():.4f} | "
                f"box {agent_box_l.item():.4f} | {elapsed:.1f}s"
            )

    # === 결과 평가 ===
    initial_loss = losses[0]
    final_loss = losses[-1]
    min_loss = min(losses)
    drop_ratio = final_loss / initial_loss if initial_loss > 0 else 0

    print("\n" + "=" * 55)
    print("Overfit Result")
    print("=" * 55)
    print(f"  Initial loss:    {initial_loss:.4f}")
    print(f"  Final loss:      {final_loss:.4f}")
    print(f"  Min loss:        {min_loss:.4f}")
    print(f"  Drop ratio:      {drop_ratio*100:.1f}%")
    print(f"  Final traj_L1:   {traj_losses[-1]:.4f} m (초기 {traj_losses[0]:.4f})")

    # 판정
    # 정상: loss가 initial의 10% 이하로 떨어짐 (완전한 overfit이면 거의 0)
    if drop_ratio > 0.5:
        print("\n  *** FAIL: loss가 절반도 안 떨어짐. 학습 파이프라인 이상 가능성 ***")
    elif drop_ratio > 0.1:
        print("\n  WARNING: loss가 떨어지긴 했지만 overfit 정도가 약함 (iter 늘리거나 lr 조정 필요)")
    else:
        print("\n  PASS: 모델이 소수 샘플을 memorize 가능 (학습 파이프라인 건강)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="v4_range",
                        help="configs/{name}.py (default: v4_range)")
    parser.add_argument("--dataroot", type=str, required=True)
    parser.add_argument("--version", type=str, default="v1.1-trainval")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3, help="Overfit용 — 정규 학습보다 높게")
    args = parser.parse_args()
    main(args)
