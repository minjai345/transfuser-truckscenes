"""
Training script for TransFuser on MAN TruckScenes.
"""

import argparse
import time

import torch
from torch.utils.data import DataLoader

from model.config import TransfuserConfig
from model.model import TransfuserModel
from model.loss import transfuser_loss
from dataset.dataset import TruckScenesDataset
from evaluate import run_evaluation

# wandb는 optional — 설치 안됐거나 --wandb 미지정이면 비활성화 상태로 동작
try:
    import wandb
except ImportError:
    wandb = None


def _format_eta(seconds: float) -> str:
    """초 단위를 h:mm:ss / m:ss로 포맷."""
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def collate_fn(batch):
    """Custom collate: stack features and targets separately."""
    features_list, targets_list = zip(*batch)

    features = {
        key: torch.stack([f[key] for f in features_list])
        for key in features_list[0]
    }
    targets = {
        key: torch.stack([t[key] for t in targets_list])
        for key in targets_list[0]
    }
    return features, targets


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Config
    config = TransfuserConfig()

    # wandb 초기화 — --wandb 플래그가 있을 때만
    use_wandb = args.wandb and wandb is not None
    if args.wandb and wandb is None:
        print("WARNING: --wandb 지정됐지만 wandb 모듈 없음. 'pip install wandb'.")
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    # Initialize TruckScenes
    from truckscenes.truckscenes import TruckScenes
    from truckscenes.utils.splits import create_splits_scenes
    print(f"Loading TruckScenes {args.version} from {args.dataroot}...")
    ts = TruckScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    # 공식 train/val scene split 사용 (scene 단위 분리 — sample 단위 분리 시 누수 위험)
    splits = create_splits_scenes()
    train_scene_names = set(splits["train"])
    val_scene_names = set(splits["val"])
    train_scene_tokens = [s["token"] for s in ts.scene if s["name"] in train_scene_names]
    val_scene_tokens = [s["token"] for s in ts.scene if s["name"] in val_scene_names]
    print(f"Split: {len(train_scene_tokens)} train scenes, {len(val_scene_tokens)} val scenes")

    # Train dataset
    train_dataset = TruckScenesDataset(
        ts=ts,
        config=config,
        num_future_samples=config.num_poses,
        split_tokens=train_scene_tokens,
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # Val dataset — run_evaluation이 인덱싱 방식으로 순회하므로 DataLoader 불필요
    val_dataset = TruckScenesDataset(
        ts=ts,
        config=config,
        num_future_samples=config.num_poses,
        split_tokens=val_scene_tokens,
    )

    # Model
    model = TransfuserModel(config=config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    print(f"Batches per epoch: {len(dataloader)}")

    # Training loop
    model.train()
    global_step = 0  # 전체 학습 스텝 카운터 (wandb x축용)
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_start = time.time()

        for batch_idx, (features, targets) in enumerate(dataloader):
            # Move to device
            features = {k: v.to(device) for k, v in features.items()}
            targets = {k: v.to(device) for k, v in targets.items()}

            # Forward
            predictions = model(features)
            loss = transfuser_loss(targets, predictions, config)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            if batch_idx % args.log_interval == 0:
                # ETA 계산: 이번 epoch 평균 배치 시간 기반
                batches_done = batch_idx + 1
                epoch_elapsed = time.time() - epoch_start
                batch_time = epoch_elapsed / batches_done
                eta_epoch = batch_time * (len(dataloader) - batches_done)
                # 전체 ETA: 남은 배치 + 남은 에폭
                remaining_batches = (len(dataloader) - batches_done) + (args.epochs - epoch - 1) * len(dataloader)
                eta_total = batch_time * remaining_batches
                print(
                    f"  Epoch {epoch+1}/{args.epochs} "
                    f"[{batch_idx}/{len(dataloader)}] "
                    f"Loss: {loss.item():.4f} | "
                    f"{batch_time:.2f}s/it | "
                    f"ETA epoch: {_format_eta(eta_epoch)} | "
                    f"total: {_format_eta(eta_total)}"
                )
                # 배치 단위 loss + lr 로깅
                if use_wandb:
                    wandb.log(
                        {
                            "train/loss": loss.item(),
                            "train/lr": optimizer.param_groups[0]["lr"],
                            "train/epoch": epoch + 1,
                        },
                        step=global_step,
                    )

        scheduler.step()
        avg_loss = epoch_loss / max(len(dataloader), 1)
        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{args.epochs} done | Avg Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | Time: {elapsed:.1f}s")

        # epoch 단위 집계 로깅
        if use_wandb:
            wandb.log(
                {
                    "epoch/avg_loss": avg_loss,
                    "epoch/lr": scheduler.get_last_lr()[0],
                    "epoch/time_sec": elapsed,
                    "epoch/index": epoch + 1,
                },
                step=global_step,
            )

        # Validation — 매 epoch 끝에 L2 + collision 평가
        eval_start = time.time()
        val_metrics = run_evaluation(
            model=model,
            ts=ts,
            dataset=val_dataset,
            config=config,
            device=device,
            ego_length=args.ego_length,
            ego_width=args.ego_width,
            log_interval=max(len(val_dataset) // 4, 1),  # 25%마다 진행 출력
            verbose=True,
        )
        eval_elapsed = time.time() - eval_start
        print(f"Eval done in {eval_elapsed:.1f}s")

        if use_wandb:
            wandb.log(
                {f"val/{k}": v for k, v in val_metrics.items()},
                step=global_step,
            )

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = f"checkpoint_epoch{epoch+1}.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "val_metrics": val_metrics,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    print("Training complete.")
    if use_wandb:
        wandb.finish()


def sanity_check(args):
    """Quick sanity check: load one sample, forward pass, compute loss."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Sanity Check (device: {device}) ===")

    config = TransfuserConfig()

    from truckscenes.truckscenes import TruckScenes
    print(f"Loading TruckScenes {args.version}...")
    ts = TruckScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    dataset = TruckScenesDataset(ts=ts, config=config, num_future_samples=config.num_poses)
    print(f"Dataset size: {len(dataset)}")

    if len(dataset) == 0:
        print("ERROR: No valid samples found!")
        return

    # Load one sample
    features, targets = dataset[0]
    print("\n--- Feature shapes ---")
    for k, v in features.items():
        print(f"  {k}: {v.shape}")
    print("\n--- Target shapes ---")
    for k, v in targets.items():
        print(f"  {k}: {v.shape}")

    # Batch dimension
    features = {k: v.unsqueeze(0).to(device) for k, v in features.items()}
    targets = {k: v.unsqueeze(0).to(device) for k, v in targets.items()}

    # Forward pass
    model = TransfuserModel(config=config).to(device)
    model.eval()
    with torch.no_grad():
        predictions = model(features)

    print("\n--- Prediction shapes ---")
    for k, v in predictions.items():
        print(f"  {k}: {v.shape}")

    # Loss
    model.train()
    predictions = model(features)
    loss = transfuser_loss(targets, predictions, config)
    print(f"\nLoss: {loss.item():.4f}")
    print(f"Loss is finite: {torch.isfinite(loss).item()}")

    # Backward
    loss.backward()
    print("Backward pass: OK")

    print("\n=== Sanity Check PASSED ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TransFuser on TruckScenes")
    parser.add_argument("--dataroot", type=str, required=True, default="./data", help="Path to TruckScenes data")
    parser.add_argument("--version", type=str, default="v1.0-mini", help="TruckScenes version")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--sanity", action="store_true", help="Run sanity check only")
    # Validation (매 epoch 끝에 실행)
    parser.add_argument("--ego_length", type=float, default=6.9, help="Ego vehicle length (m)")
    parser.add_argument("--ego_width", type=float, default=2.5, help="Ego vehicle width (m)")
    # wandb 관련 플래그
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="transfuser-truckscenes")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Run name (default: wandb auto)")

    args = parser.parse_args()

    if args.sanity:
        sanity_check(args)
    else:
        train(args)


