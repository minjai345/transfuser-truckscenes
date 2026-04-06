"""
Training script for TransFuser on MAN TruckScenes.
"""

import argparse
import time

import torch
from torch.utils.data import DataLoader

from transfuser.transfuser_config import TransfuserConfig
from transfuser.transfuser_model import TransfuserModel
from transfuser.transfuser_loss import transfuser_loss
from transfuser.dataset import TruckScenesDataset


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

    # Initialize TruckScenes
    from truckscenes.truckscenes import TruckScenes
    print(f"Loading TruckScenes {args.version} from {args.dataroot}...")
    ts = TruckScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    # Dataset & DataLoader
    dataset = TruckScenesDataset(
        ts=ts,
        config=config,
        num_future_samples=config.num_poses,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # Model
    model = TransfuserModel(config=config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Batches per epoch: {len(dataloader)}")

    # Training loop
    model.train()
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

            if batch_idx % args.log_interval == 0:
                print(
                    f"  Epoch {epoch+1}/{args.epochs} "
                    f"[{batch_idx}/{len(dataloader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = epoch_loss / max(len(dataloader), 1)
        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{args.epochs} done | Avg Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = f"checkpoint_epoch{epoch+1}.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    print("Training complete.")


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
    parser.add_argument("--dataroot", type=str, required=True, help="Path to TruckScenes data")
    parser.add_argument("--version", type=str, default="v1.0-mini", help="TruckScenes version")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--sanity", action="store_true", help="Run sanity check only")

    args = parser.parse_args()

    if args.sanity:
        sanity_check(args)
    else:
        train(args)


