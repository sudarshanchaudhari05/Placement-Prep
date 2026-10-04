"""
Perception Model Training Script (50-Epoch Production Pipeline)
==============================================================
Trains RowAnchorNet (ShuffleNetV2-0.5x backbone) on the 6,000-sample synthetic dataset.
- Training set: data/synthetic/train/ (4,200 images)
- Validation set: data/synthetic/val/ (900 images)
- Checkpoints: saved to experiments/<run_name>/checkpoints/
- Logs: saved to experiments/<run_name>/training_log.csv & json

FROZEN RESEARCH SPECIFICATION:
- Input: 640x480 RGB -> SICT -> ROI[200:470, 0:640] -> Resize[288, 384] -> Tensor
- Model: ShuffleNetV2-0.5x ImageNet pretrained backbone, M=18 row anchors, K=100 spatial bins
- Loss: L_total = L_focal (gamma=2.0) + 0.5 * L_smooth + 0.75 * L_curv
- Optimizer: AdamW (lr=4e-3, betas=(0.9, 0.999), weight_decay=1e-4)
- Scheduler: CosineAnnealingLR (T_max=50, eta_min=1e-5)
- Batch size: 32
- Master seed: 42
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import load_config, ProjectConfig
from src.data.dataset import RowAnchorDataset, build_dataloader
from src.models.row_anchor_net import build_model
from src.losses.structural_loss import TotalRowAnchorLoss
from src.engine.trainer import build_optimizer, build_scheduler, train_one_step
from src.engine.validator import validate_one_step
from src.evaluation.metrics import RowAnchorEvaluator
from src.utils.checkpoint import save_checkpoint
from src.utils.seed import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Train RowAnchorNet on Synthetic Dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Total training epochs (default: 50)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=4.0e-3, help="Initial learning rate (default: 4e-3)")
    parser.add_argument("--final_lr", type=float, default=1.0e-5, help="Final learning rate (default: 1e-5)")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed (default: 42)")
    parser.add_argument("--device", type=str, default="cuda", help="Target device: 'cuda' or 'cpu'")
    parser.add_argument("--run_name", type=str, default="shufflenet_v2_x0_5_baseline", help="Run identifier")
    parser.add_argument("--use_amp", action="store_true", default=False, help="Enable CUDA AMP (default: False)")
    parser.add_argument("--dry_run", action="store_true", default=False, help="Debug dry run on small subset")
    return parser.parse_args()


def train_pipeline(args):
    # Enforce reproducibility
    seed_everything(args.seed, deterministic_cuda=True)

    # Load configuration
    cfg = load_config()
    cfg.training.max_epochs = args.epochs
    cfg.training.batch_size = args.batch_size
    cfg.training.initial_lr = args.lr
    cfg.training.final_lr = args.final_lr
    cfg.experiment.seed = args.seed
    cfg.validate()

    # Hardware setup
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"[Device] Using {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # Paths
    run_dir = PROJECT_ROOT / "experiments" / args.run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_ann = PROJECT_ROOT / "data" / "synthetic" / "train" / "annotations.json"
    val_ann = PROJECT_ROOT / "data" / "synthetic" / "val" / "annotations.json"

    assert train_ann.exists(), f"Train annotations missing: {train_ann}"
    assert val_ann.exists(), f"Val annotations missing: {val_ann}"

    # Datasets and Loaders
    print("[Data] Initializing Datasets and DataLoaders...")
    train_ds = RowAnchorDataset(annotation_file=train_ann, config=cfg)
    val_ds = RowAnchorDataset(annotation_file=val_ann, config=cfg)

    g_train = torch.Generator()
    g_train.manual_seed(args.seed)

    train_loader = build_dataloader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        generator=g_train,
    )
    val_loader = build_dataloader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )
    print(f"  Train: {len(train_ds)} samples ({len(train_loader)} batches/epoch)")
    print(f"  Val:   {len(val_ds)} samples ({len(val_loader)} batches/epoch)")

    # Model
    print("[Model] Building RowAnchorNet (ShuffleNetV2-0.5x backbone)...")
    model = build_model(cfg, apply_sict_in_forward=False).to(device)
    params = model.count_parameters()
    print(f"  Total Parameters: {params['total']:,} (Trainable: {params['trainable']:,})")

    # Loss & Optimizer
    criterion = TotalRowAnchorLoss(loss_config=cfg.loss, model_config=cfg.model).to(device)
    optimizer = build_optimizer(model, cfg.training)
    scheduler = build_scheduler(optimizer, cfg.training, total_epochs=args.epochs)
    evaluator = RowAnchorEvaluator(cfg)

    # Optional AMP scaler
    scaler = torch.amp.GradScaler("cuda", enabled=args.use_amp) if device.type == "cuda" else None

    # Training state
    best_val_loss = float("inf")
    history = []

    print("\n" + "=" * 75)
    print(f"STARTING TRAINING: {args.epochs} EPOCHS, BATCH SIZE {args.batch_size}, LR {args.lr}")
    print("=" * 75)

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        train_losses = []
        train_focal = []
        train_smooth = []
        train_curv = []

        for batch_idx, batch in enumerate(train_loader):
            if args.dry_run and batch_idx >= 2:
                break

            images = batch["image"].to(device)
            targets = batch["targets"].to(device)

            optimizer.zero_grad()
            if args.use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    logits = model(images)
                    loss_dict = criterion(logits, targets)
                scaler.scale(loss_dict["loss_total"]).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss_dict = criterion(logits, targets)
                loss_dict["loss_total"].backward()
                optimizer.step()

            train_losses.append(loss_dict["loss_total"].item())
            train_focal.append(loss_dict["loss_focal"].item())
            train_smooth.append(loss_dict["loss_smooth"].item())
            train_curv.append(loss_dict["loss_curv"].item())

        # Update learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # Validation phase
        model.eval()
        val_losses = []
        val_focal = []
        val_smooth = []
        val_curv = []
        evaluator.reset()

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if args.dry_run and batch_idx >= 2:
                    break

                images = batch["image"].to(device)
                targets = batch["targets"].to(device)
                u_coords = batch["u_coords"]
                presence = batch["presence"]

                logits = model(images)
                loss_dict = criterion(logits, targets)

                val_losses.append(loss_dict["loss_total"].item())
                val_focal.append(loss_dict["loss_focal"].item())
                val_smooth.append(loss_dict["loss_smooth"].item())
                val_curv.append(loss_dict["loss_curv"].item())
                evaluator.update(logits.cpu(), u_coords, presence)

        epoch_metrics = evaluator.compute()
        epoch_duration = time.time() - epoch_start

        avg_train_loss = float(np.mean(train_losses))
        avg_val_loss = float(np.mean(val_losses))

        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss

        # Checkpoint save
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_metrics": epoch_metrics,
            "learning_rate": current_lr,
            "seed": args.seed,
            "config": cfg.to_dict(),
        }
        save_checkpoint(
            state,
            checkpoint_dir=ckpt_dir,
            filename=f"checkpoint_epoch_{epoch:03d}.pth",
            is_best=is_best,
            config=cfg,
        )
        # Always maintain latest checkpoint link
        save_checkpoint(
            state,
            checkpoint_dir=ckpt_dir,
            filename="checkpoint.pth",
            is_best=False,
            config=None,
        )

        record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_f1": epoch_metrics.get("f1_score", 0.0),
            "val_precision": epoch_metrics.get("precision", 0.0),
            "val_recall": epoch_metrics.get("recall", 0.0),
            "val_rmse": epoch_metrics.get("rmse_overall", 0.0),
            "lr": current_lr,
            "duration_sec": epoch_duration,
            "is_best": is_best,
        }
        history.append(record)

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val F1: {epoch_metrics.get('f1_score', 0.0):.4f} | "
            f"Val RMSE: {epoch_metrics.get('rmse_overall', 0.0):.2f}px | "
            f"LR: {current_lr:.2e} | "
            f"Time: {epoch_duration:.1f}s {'[*BEST*]' if is_best else ''}"
        )

    total_training_time = time.time() - start_time
    print("\n" + "=" * 75)
    print(f"TRAINING COMPLETED: {total_training_time/60:.2f} min total")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {ckpt_dir}")
    print("=" * 75)

    # Save training history JSON
    history_file = run_dir / "training_history.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump({"history": history, "total_time_sec": total_training_time, "best_val_loss": best_val_loss}, f, indent=2)

    return history


if __name__ == "__main__":
    cli_args = parse_args()
    train_pipeline(cli_args)
