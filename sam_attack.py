"""Generate a SAM attack and optionally evaluate transfer to other SAM models."""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from sam_attacks import SamLogits, attack_sam


def get_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=["vit_b", "vit_l", "vit_h"], default="vit_b")
    parser.add_argument("--point", nargs=3, type=float, action="append", metavar=("X", "Y", "LABEL"),
                        help="point in original-image pixels; label 1=foreground, 0=background; repeatable")
    parser.add_argument("--box", nargs=4, type=float, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--attack", choices=["ni", "ni_di_ti"], default="ni")
    parser.add_argument("--epsilon", type=int, default=12, help="L-infinity budget in 0-255 pixel units")
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("results/sam"))
    parser.add_argument("--target", action="append", default=[], metavar="MODEL=CHECKPOINT",
                        help="evaluate transfer on vit_l/vit_h/etc.; repeatable, e.g. vit_l=checkpoints/sam_vit_l.pth")
    return parser


def mask_iou(first, second):
    intersection = (first & second).sum().item()
    union = (first | second).sum().item()
    return intersection / union if union else 1.0


def save_mask(path, mask):
    pixels = mask[0, 0].detach().cpu().numpy().astype(np.uint8) * 255
    Image.fromarray(pixels).save(path)


def load_model(model_type, checkpoint, device):
    try:
        from segment_anything import sam_model_registry
    except ImportError as error:
        raise RuntimeError("install the optional SAM dependencies: pip install -r requirements-sam.txt") from error
    return sam_model_registry[model_type](checkpoint=str(checkpoint)).to(device)


def run(args):
    if not args.point and args.box is None:
        raise ValueError("provide --point X Y LABEL or --box X0 Y0 X1 Y1")
    if not 0 <= args.epsilon <= 255:
        raise ValueError("--epsilon must be in [0, 255]")
    if args.iterations < 1:
        raise ValueError("--iterations must be positive")
    if not args.image.is_file() or not args.checkpoint.is_file():
        raise ValueError("--image and --checkpoint must be existing files")
    targets = []
    for specification in args.target:
        model_type, separator, checkpoint = specification.partition("=")
        if not separator or model_type not in {"vit_b", "vit_l", "vit_h"} or not Path(checkpoint).is_file():
            raise ValueError("each --target must be MODEL=CHECKPOINT with an existing SAM checkpoint")
        targets.append((model_type, Path(checkpoint)))
    torch.manual_seed(args.seed)
    pixels = np.array(Image.open(args.image).convert("RGB"), copy=True)
    clean = torch.from_numpy(pixels).permute(2, 0, 1)[None].to(args.device, dtype=torch.float32) / 255
    points = None if not args.point else [point[:2] for point in args.point]
    labels = None if not args.point else [point[2] for point in args.point]
    sam = load_model(args.model, args.checkpoint, args.device)
    adapter = SamLogits(sam, pixels.shape[:2], points, labels, args.box)
    result = attack_sam(adapter, clean, attack=args.attack, epsilon=args.epsilon / 255,
                        iterations=args.iterations, seed=args.seed)
    # Evaluate the same 8-bit pixels that will be saved, including quantization.
    adv_pixels = (result.adversarial[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    adversarial = torch.from_numpy(adv_pixels).permute(2, 0, 1)[None].to(args.device, dtype=torch.float32) / 255
    with torch.no_grad():
        adv_logits = adapter(adversarial)
        clean_mask = result.pseudo_labels.bool()
        adv_mask = adv_logits > sam.mask_threshold
        saved_loss = F.binary_cross_entropy_with_logits(adv_logits, result.pseudo_labels).item()
    max_delta = int(np.abs(adv_pixels.astype(np.int16) - pixels.astype(np.int16)).max())
    if max_delta > args.epsilon:
        raise RuntimeError("saved image exceeds the requested pixel budget")
    args.output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(adv_pixels).save(args.output / "adversarial.png")
    save_mask(args.output / "source_clean_mask.png", clean_mask)
    save_mask(args.output / "source_adversarial_mask.png", adv_mask)
    report = {
        "protocol": "fixed_clean_pseudo_label_bce",
        "source_model": args.model, "source_checkpoint": str(args.checkpoint.resolve()),
        "image": str(args.image.resolve()), "image_size_hw": list(pixels.shape[:2]),
        "attack": args.attack, "epsilon_pixel_units": args.epsilon, "iterations": args.iterations,
        "seed": args.seed, "device": args.device, "points_xy_label": args.point, "box_xyxy": args.box,
        "multimask_output": False, "mask_threshold": float(sam.mask_threshold),
        "max_saved_pixel_delta": max_delta,
        "source": {"clean_bce": result.clean_loss, "adversarial_bce": saved_loss,
                   "clean_adversarial_mask_iou": mask_iou(clean_mask, adv_mask),
                   "clean_foreground_pixels": int(clean_mask.sum().item()),
                   "adversarial_foreground_pixels": int(adv_mask.sum().item())},
        "targets": [], "torch_version": torch.__version__,
    }
    # Load transfer targets sequentially; they never contribute attack gradients
    # or the pseudo-label used during optimization.
    del adapter, sam, result, adv_logits
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    for index, (model_type, checkpoint) in enumerate(targets):
        sam = load_model(model_type, checkpoint, args.device)
        adapter = SamLogits(sam, pixels.shape[:2], points, labels, args.box)
        with torch.no_grad():
            target_clean = adapter(clean) > sam.mask_threshold
            target_adv = adapter(adversarial) > sam.mask_threshold
        save_mask(args.output / f"target_{index}_{model_type}_clean_mask.png", target_clean)
        save_mask(args.output / f"target_{index}_{model_type}_adversarial_mask.png", target_adv)
        report["targets"].append({
            "model": model_type, "checkpoint": str(checkpoint.resolve()),
            "clean_adversarial_mask_iou": mask_iou(target_clean, target_adv),
            "clean_foreground_pixels": int(target_clean.sum().item()),
            "adversarial_foreground_pixels": int(target_adv.sum().item()),
        })
        del adapter, sam
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = get_argparser()
    args = parser.parse_args()
    try:
        report = run(args)
    except (ValueError, RuntimeError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
