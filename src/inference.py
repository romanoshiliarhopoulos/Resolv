"""
Run denoising inference on one or more images.

Usage:
    poetry run python -m src.inference input.png
    poetry run python -m src.inference data/renders/noisy_0002spp/
    poetry run python -m src.inference input.png --checkpoint checkpoints/best.pt
    poetry run python -m src.inference input.png --out results/
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from src.model.unet import UNet


def load_model(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(checkpoint, map_location=device)
    model = UNet()
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model


@torch.inference_mode()
def denoise(model: torch.nn.Module, image_path: Path, device: torch.device) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    x = TF.to_tensor(img).unsqueeze(0).to(device)
    out = model(x).squeeze(0).clamp(0, 1).cpu()
    return TF.to_pil_image(out)


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    print(f"checkpoint: {checkpoint}")

    model = load_model(checkpoint, device)

    input_path = Path(args.input)
    if input_path.is_dir():
        images = sorted(input_path.glob("*.png")) + sorted(input_path.glob("*.jpg"))
    else:
        images = [input_path]

    if not images:
        raise FileNotFoundError(f"No images found at {input_path}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        result = denoise(model, img_path, device)
        out_path = out_dir / img_path.name
        result.save(out_path)
        print(f"  {img_path.name} -> {out_path}")

    print(f"done — {len(images)} image(s) saved to {out_dir}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Denoise images using a trained UNet checkpoint.")
    p.add_argument("input",        help="Path to a noisy image or directory of images")
    p.add_argument("--checkpoint", default="checkpoints/best.pt", help="Model checkpoint to use")
    p.add_argument("--out",        default="inference",           help="Output directory")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse_args())
