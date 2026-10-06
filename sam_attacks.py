"""Label-free, bounded attacks on SAM using fixed clean-prediction masks.

This is a new implementation of the pseudo-label protocol, not a recovered
script for the historical SAM figures. Images are RGB tensors in [0, 1].
"""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


class SamLogits(nn.Module):
    """Differentiable SAM path for one image and one point/box prompt set.

    Call the image encoder and mask decoder directly: Sam.forward and
    SamPredictor's public inference methods disable gradient tracking.
    Use the single-mask decoder output consistently for clean and attacked
    images, rather than treating candidate masks as semantic classes.
    """

    def __init__(self, sam, original_size, point_coords=None, point_labels=None,
                 box=None):
        super().__init__()
        self.sam = sam.eval().requires_grad_(False)
        self.original_size = tuple(original_size)
        if len(self.original_size) != 2 or min(self.original_size) <= 0:
            raise ValueError("original_size must contain positive (height, width)")
        height, width = self.original_size
        device = sam.device
        points = labels = rectangle = None
        if point_coords is not None:
            points = torch.as_tensor(point_coords, dtype=torch.float32, device=device)
            if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
                raise ValueError("point_coords must have shape N x 2 in (x, y) order")
            if point_labels is None:
                point_labels = [1] * points.shape[0]
            labels = torch.as_tensor(point_labels, device=device)
            if labels.shape != (points.shape[0],) or not torch.all((labels == 0) | (labels == 1)):
                raise ValueError("point_labels must contain one 0/1 label per point")
            labels = labels.to(torch.int64)
            if not torch.isfinite(points).all() or torch.any(points < 0) or torch.any(points >= points.new_tensor([width, height])):
                raise ValueError("point coordinates must lie inside the original image")
        elif point_labels is not None:
            raise ValueError("point_labels require point_coords")
        if box is not None:
            rectangle = torch.as_tensor(box, dtype=torch.float32, device=device)
            if rectangle.shape != (4,) or not torch.isfinite(rectangle).all():
                raise ValueError("box must be finite (x0, y0, x1, y1)")
            x0, y0, x1, y1 = rectangle.tolist()
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise ValueError("box must have positive area inside the original image")
        if points is None and rectangle is None:
            raise ValueError("provide at least one point or a box prompt")
        self.register_buffer("point_coords", None if points is None else points.detach().clone())
        self.register_buffer("point_labels", None if labels is None else labels.detach().clone())
        self.register_buffer("box", None if rectangle is None else rectangle.detach().clone())

    def forward(self, image, point_coords=None, box=None):
        if image.shape != (1, 3, *self.original_size):
            raise ValueError("image must have shape 1 x 3 x H x W for original_size")
        height, width = self.original_size
        side = self.sam.image_encoder.img_size
        scale = side / max(height, width)
        resized_size = (int(height * scale + 0.5), int(width * scale + 0.5))
        resized = F.interpolate(image, resized_size, mode="bilinear",
                                align_corners=False, antialias=True)
        image_embedding = self.sam.image_encoder(self.sam.preprocess(resized[0] * 255)[None])
        points = self.point_coords if point_coords is None else point_coords
        rectangle = self.box if box is None else box
        coord_scale = image.new_tensor([resized_size[1] / width, resized_size[0] / height])
        # Prompts are fixed conditioning inputs; only the image needs gradients.
        with torch.no_grad():
            prompt = None if points is None else ((points * coord_scale)[None], self.point_labels[None])
            boxes = None if rectangle is None else (rectangle.reshape(2, 2) * coord_scale).reshape(1, 4)
            sparse, dense = self.sam.prompt_encoder(points=prompt, boxes=boxes, masks=None)
            image_pe = self.sam.prompt_encoder.get_dense_pe()
        logits, _ = self.sam.mask_decoder(
            image_embeddings=image_embedding, image_pe=image_pe,
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        return self.sam.postprocess_masks(logits, resized_size, self.original_size)


@dataclass
class AttackResult:
    adversarial: torch.Tensor
    pseudo_labels: torch.Tensor
    clean_logits: torch.Tensor
    clean_loss: float
    adversarial_loss: float


def _diverse_view(image, labels, points, box, generator):
    """Resize/pad a view, applying identical geometry to labels and prompts."""
    height, width = image.shape[-2:]
    scale = 0.9 + 0.1 * torch.rand((), generator=generator).item()
    new_h, new_w = max(1, round(height * scale)), max(1, round(width * scale))
    top = torch.randint(height - new_h + 1, (), generator=generator).item()
    left = torch.randint(width - new_w + 1, (), generator=generator).item()
    padding = (left, width - new_w - left, top, height - new_h - top)
    view = F.pad(F.interpolate(image, (new_h, new_w), mode="bilinear", align_corners=False, antialias=True), padding)
    target = F.pad(F.interpolate(labels, (new_h, new_w), mode="nearest"), padding)
    valid = F.pad(torch.ones_like(target[..., :new_h, :new_w]), padding)
    coord_scale = image.new_tensor([new_w / width, new_h / height])
    offset = image.new_tensor([left, top])
    points = None if points is None else points * coord_scale + offset
    box = None if box is None else (box.reshape(2, 2) * coord_scale + offset).reshape(4)
    return view, target, valid, points, box


def _smooth_gradient(gradient):
    axis = torch.arange(-2, 3, dtype=gradient.dtype, device=gradient.device)
    kernel_1d = torch.exp(-0.5 * (axis / 3.0).square())
    kernel = kernel_1d[:, None] * kernel_1d[None, :]
    kernel = (kernel / kernel.sum())[None, None].expand(3, 1, 5, 5)
    return F.conv2d(gradient, kernel, padding=2, groups=3)


@torch.enable_grad()
def attack_sam(model, image, *, attack="ni", epsilon=12 / 255,
               iterations=16, diversity_probability=0.7, seed=0):
    """Maximize BCE against one detached, clean-source pseudo-label mask.

    NI uses Nesterov lookahead and normalized momentum. NI_DI_TI additionally
    applies aligned resize/pad views and Gaussian smoothing of image gradients.
    Every update is projected onto both [0, 1] and the clean-image L-infinity ball.
    """
    if attack not in {"ni", "ni_di_ti"}:
        raise ValueError("attack must be ni or ni_di_ti")
    if not math.isfinite(epsilon) or not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be finite and in [0, 1]")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if not 0 <= diversity_probability <= 1:
        raise ValueError("diversity_probability must be in [0, 1]")
    if not image.is_floating_point() or not torch.isfinite(image).all() or torch.any(image < 0) or torch.any(image > 1):
        raise ValueError("image must be finite floating-point RGB in [0, 1]")
    clean = image.detach().clone()
    with torch.no_grad():
        clean_logits = model(clean).detach()
        target = (clean_logits > model.sam.mask_threshold).to(clean_logits.dtype).detach()
        clean_loss = F.binary_cross_entropy_with_logits(clean_logits, target).item()
    adversarial = clean.clone()
    momentum = torch.zeros_like(clean)
    step = epsilon / iterations
    generator = torch.Generator().manual_seed(seed)
    lower, upper = (clean - epsilon).clamp(0, 1), (clean + epsilon).clamp(0, 1)
    for _ in range(iterations if epsilon else 0):
        lookahead = (adversarial + step * momentum).clamp(0, 1).detach().requires_grad_(True)
        if attack == "ni_di_ti" and torch.rand((), generator=generator).item() < diversity_probability:
            view, view_target, valid, points, box = _diverse_view(
                lookahead, target, model.point_coords, model.box, generator)
            logits = model(view, point_coords=points, box=box)
            pixel_loss = F.binary_cross_entropy_with_logits(logits, view_target, reduction="none")
            loss = (pixel_loss * valid).sum() / valid.sum()
        else:
            loss = F.binary_cross_entropy_with_logits(model(lookahead), target)
        gradient, = torch.autograd.grad(loss, lookahead)
        if not torch.isfinite(gradient).all():
            raise RuntimeError("SAM produced a non-finite input gradient")
        if attack == "ni_di_ti":
            gradient = _smooth_gradient(gradient)
        norm = gradient.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(torch.finfo(gradient.dtype).eps)
        momentum = (momentum + gradient / norm).detach()
        adversarial = torch.maximum(torch.minimum(adversarial + step * momentum.sign(), upper), lower).detach()
    with torch.no_grad():
        adversarial_loss = F.binary_cross_entropy_with_logits(model(adversarial), target).item()
    return AttackResult(adversarial, target, clean_logits, clean_loss, adversarial_loss)
