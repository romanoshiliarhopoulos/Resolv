"""
Training orchestrator for image-to-image denoising models.

Usage:
    poetry run python -m src.train [options]

Examples:
    # Quick sanity check — overfit 10 pairs
    poetry run python -m src.train --renders data/renders --overfit 10 --epochs 200

    # Full training run
    poetry run python -m src.train --renders data/renders --epochs 100 --batch-size 4

    # Custom model checkpoint + resume
    poetry run python -m src.train --renders data/renders --resume checkpoints/last.pt
"""

import argparse
import importlib

from tqdm import trange, tqdm
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import functional as TF
from PIL import Image


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DenoiseDataset(Dataset):
    """
    Pairs every noisy image with its clean counterpart by matching filename stems.
    Supports multiple noisy subdirectories (one per spp level).
    """

    def __init__(self, renders_dir: Path, noise_dirs: list[str], patch_size: int | None = None):
        self.patch_size = patch_size
        clean_dir = renders_dir / "clean"

        self.pairs: list[tuple[Path, Path]] = []
        for nd in noise_dirs:
            noisy_dir = renders_dir / nd
            if not noisy_dir.exists():
                continue
            for noisy_path in sorted(noisy_dir.glob("*.png")):
                clean_path = clean_dir / noisy_path.name
                if clean_path.exists():
                    self.pairs.append((noisy_path, clean_path))

        if not self.pairs:
            raise FileNotFoundError(
                f"No matched noisy/clean pairs found under {renders_dir}. "
                f"Looked for noise dirs: {noise_dirs}"
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        noisy_path, clean_path = self.pairs[idx]
        noisy = _load(noisy_path)
        clean = _load(clean_path)

        if self.patch_size:
            noisy, clean = _random_crop(noisy, clean, self.patch_size)
            if torch.rand(1).item() > 0.5:
                noisy, clean = TF.hflip(noisy), TF.hflip(clean)

        return noisy, clean


def _load(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return TF.to_tensor(img)  # [0, 1] float32, shape (3, H, W)


def _random_crop(a: torch.Tensor, b: torch.Tensor, size: int):
    _, h, w = a.shape
    top = torch.randint(0, h - size + 1, (1,)).item()
    left = torch.randint(0, w - size + 1, (1,)).item()
    return TF.crop(a, top, left, size, size), TF.crop(b, top, left, size, size)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def load_model(model_spec: str, **kwargs) -> nn.Module:
    """
    Load a model by dotted path, e.g. 'src.model.unet.UNet'.
    Extra kwargs are forwarded to the constructor.
    """
    module_path, class_name = model_spec.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # --- dataset ---
    renders_dir = Path(args.renders)
    noise_dirs = args.noise_dirs.split(",")
    dataset = DenoiseDataset(renders_dir, noise_dirs, patch_size=args.patch_size)
    print(f"dataset: {len(dataset)} pairs from {renders_dir}")

    if args.overfit:
        n = min(args.overfit, len(dataset))
        dataset.pairs = dataset.pairs[:n]
        train_ds, val_ds = dataset, dataset
        print(f"overfit mode: using {n} pairs for both train and val")
    else:
        val_n = max(1, int(len(dataset) * args.val_split))
        train_n = len(dataset) - val_n
        train_ds, val_ds = random_split(dataset, [train_n, val_n])
        print(f"train: {train_n}  val: {val_n}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    # --- model ---
    model = load_model(args.model, **_parse_model_kwargs(args.model_kwargs))
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model: {args.model}  ({total_params:.2f}M params)")

    if args.checkpoint_activations:
        model.use_checkpointing()

    # --- loss ---
    loss_fn = _build_loss(args.loss).to(device)

    # --- optimizer & scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr / 100)
    scaler    = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_epoch = 1
    best_val = float("inf")

    # --- resume ---
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", best_val)
        print(f"resumed from {args.resume} (epoch {ckpt['epoch']})")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- loop ---
    epoch_bar = trange(start_epoch, args.epochs + 1, desc="training", unit="epoch")
    for epoch in epoch_bar:
        train_loss = _run_epoch(model, train_loader, loss_fn, optimizer, scaler, device, train=True)
        val_loss   = _run_epoch(model, val_loader,   loss_fn, None,      None,   device, train=False)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        epoch_bar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}", lr=f"{lr:.2e}")

        _save(ckpt_dir / "last.pt", model, optimizer, scheduler, epoch, best_val)
        if val_loss < best_val:
            best_val = val_loss
            _save(ckpt_dir / "best.pt", model, optimizer, scheduler, epoch, best_val)
            tqdm.write(f"epoch {epoch:4d}  new best val {best_val:.4f}")


def _run_epoch(model, loader, loss_fn, optimizer, scaler, device, train: bool) -> float:
    model.train(train)
    total = 0.0
    phase = "train" if train else "val"
    with torch.set_grad_enabled(train):
        batch_bar = tqdm(loader, desc=f"  {phase}", leave=False, unit="batch")
        for noisy, clean in batch_bar:
            noisy, clean = noisy.to(device), clean.to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                pred = model(noisy)
                loss = loss_fn(pred, clean)
            if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            total += loss.item()
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")
    return total / len(loader)


def _save(path, model, optimizer, scheduler, epoch, best_val):
    torch.save({
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch":     epoch,
        "best_val":  best_val,
    }, path)


def _build_loss(name: str) -> nn.Module:
    if name == "denoising":
        from src.model.loss import DenoiseLoss
        return DenoiseLoss()
    if name == "l1":
        return nn.L1Loss()
    if name == "mse":
        return nn.MSELoss()
    # dotted path to a custom loss class
    return load_model(name)


def _parse_model_kwargs(raw: str) -> dict:
    if not raw:
        return {}
    kwargs = {}
    for pair in raw.split(","):
        k, v = pair.split("=")
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                if v.lower() in ("true", "false"):
                    v = v.lower() == "true"
        kwargs[k] = v
    return kwargs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train an image denoising model.")

    # paths
    p.add_argument("--renders",        default="data/renders",       help="Root renders directory")
    p.add_argument("--noise-dirs",     default="noisy_0002spp,noisy_0004spp,noisy_0016spp,noisy_0064spp",
                                                                      help="Comma-separated noisy subdirs to use")
    p.add_argument("--checkpoint-dir", default="checkpoints",        help="Where to save checkpoints")
    p.add_argument("--resume",         default=None,                 help="Path to checkpoint to resume from")

    # model
    p.add_argument("--model",          default="src.model.unet.UNet",help="Dotted path to model class")
    p.add_argument("--model-kwargs",   default="",                   help="Constructor kwargs, e.g. base_channels=32")
    p.add_argument("--checkpoint-activations", action="store_true",  help="Enable gradient checkpointing")

    # loss
    p.add_argument("--loss",           default="denoising",          help="Loss: denoising | l1 | mse | dotted.path.Class")

    # training
    p.add_argument("--epochs",         type=int,   default=100)
    p.add_argument("--batch-size",     type=int,   default=4)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--val-split",      type=float, default=0.2,      help="Fraction of data for validation")
    p.add_argument("--patch-size",     type=int,   default=256,      help="Random crop size (None = full image)")
    p.add_argument("--workers",        type=int,   default=4)

    # debug
    p.add_argument("--overfit",        type=int,   default=None,     help="Overfit on N samples (no val split)")

    return p.parse_args()


if __name__ == "__main__":
    train(_parse_args())
