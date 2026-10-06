"""CPU checks for the attack objective, geometry, bounds, and official SAM API."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from sam_attacks import SamLogits, _diverse_view, attack_sam
from sam_attack import get_argparser, run


class ToyPromptEncoder(nn.Module):
    def forward(self, points, boxes, masks):
        self.last_points, self.last_boxes = points, boxes
        return torch.zeros(1, 1, 3), torch.zeros(1, 3, 16, 16)

    def get_dense_pe(self):
        return torch.zeros(1, 3, 16, 16)


class ToyImageEncoder(nn.Identity):
    img_size = 16


class ToyMaskDecoder(nn.Module):
    def forward(self, image_embeddings, **kwargs):
        logits = 8 * (image_embeddings.mean(1, keepdim=True) - 0.5)
        return logits, logits.new_zeros(1, 1)


class ToySam(nn.Module):
    mask_threshold = 0.0

    def __init__(self):
        super().__init__()
        self.image_encoder = ToyImageEncoder()
        self.prompt_encoder = ToyPromptEncoder()
        self.mask_decoder = ToyMaskDecoder()

    @property
    def device(self):
        return torch.device("cpu")

    def preprocess(self, image):
        return F.pad(image / 255, (0, 16 - image.shape[-1], 0, 16 - image.shape[-2]))

    def postprocess_masks(self, masks, input_size, original_size):
        return F.interpolate(masks[..., :input_size[0], :input_size[1]], original_size,
                             mode="bilinear", align_corners=False)


class SamAttackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.image = torch.linspace(0.15, 0.85, 200).reshape(1, 1, 10, 20).repeat(1, 3, 1, 1)
        self.adapter = SamLogits(ToySam(), (10, 20), [[5, 4]], [1], [1, 1, 18, 9])

    def test_bce_matches_two_class_cross_entropy_and_gradients(self):
        logits = torch.tensor([[[[-2.0, 0.5, 3.0]]]], requires_grad=True)
        labels = torch.tensor([[[[0.0, 1.0, 1.0]]]])
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        ce = F.cross_entropy(torch.cat([torch.zeros_like(logits), logits], dim=1), labels[:, 0].long())
        torch.testing.assert_close(bce, ce)
        torch.testing.assert_close(torch.autograd.grad(bce, logits)[0], torch.autograd.grad(ce, logits)[0])

    def test_ni_increases_fixed_pseudo_label_loss_within_budget(self):
        original = self.image.clone()
        result = attack_sam(self.adapter, self.image, epsilon=12 / 255, iterations=16)
        torch.testing.assert_close(self.image, original)
        self.assertTrue(torch.equal(result.pseudo_labels, (result.clean_logits > 0).float()))
        self.assertFalse(result.pseudo_labels.requires_grad)
        self.assertFalse(result.adversarial.requires_grad)
        self.assertGreater(result.adversarial_loss, result.clean_loss)
        self.assertGreater((result.adversarial - original).abs().max().item(), 0)
        self.assertLessEqual((result.adversarial - original).abs().max().item(), 12 / 255 + 1e-7)
        self.assertTrue(torch.all((result.adversarial >= 0) & (result.adversarial <= 1)))

    def test_ensemble_is_bounded_and_deterministic_with_diversity(self):
        options = dict(attack="ni_di_ti", epsilon=0.05, iterations=8, diversity_probability=1.0, seed=42)
        first = attack_sam(self.adapter, self.image, **options)
        second = attack_sam(self.adapter, self.image, **options)
        torch.testing.assert_close(first.adversarial, second.adversarial)
        self.assertGreater(first.adversarial_loss, first.clean_loss)
        self.assertLessEqual((first.adversarial - self.image).abs().max().item(), 0.05 + 1e-7)

    def test_diversity_aligns_prompts_masks_and_valid_pixels(self):
        labels = torch.ones(1, 1, 10, 20)
        view, target, valid, points, box = _diverse_view(
            self.image, labels, self.adapter.point_coords, self.adapter.box,
            torch.Generator().manual_seed(7))
        self.assertEqual(view.shape, self.image.shape)
        self.assertTrue(torch.equal(target, valid))
        rows, cols = torch.where(valid[0, 0] > 0)
        new_h, new_w = int(rows.max() - rows.min() + 1), int(cols.max() - cols.min() + 1)
        scale = torch.tensor([new_w / 20, new_h / 10])
        offset = torch.tensor([cols.min(), rows.min()])
        torch.testing.assert_close(points, self.adapter.point_coords * scale + offset)
        torch.testing.assert_close(box.reshape(2, 2), self.adapter.box.reshape(2, 2) * scale + offset)

    def test_adapter_rescales_points_and_boxes_to_encoder_frame(self):
        self.adapter(self.image)
        points, labels = self.adapter.sam.prompt_encoder.last_points
        torch.testing.assert_close(points, torch.tensor([[[4.0, 3.2]]]))
        torch.testing.assert_close(labels, torch.tensor([[1]]))
        torch.testing.assert_close(self.adapter.sam.prompt_encoder.last_boxes, torch.tensor([[0.8, 0.8, 14.4, 7.2]]))

    def test_zero_budget_and_zero_gradient_remain_finite(self):
        result = attack_sam(self.adapter, self.image, epsilon=0)
        torch.testing.assert_close(result.adversarial, self.image)
        with patch.object(self.adapter, "forward", side_effect=lambda image, **kw: image[:, :1] * 0):
            result = attack_sam(self.adapter, self.image, epsilon=0.05, iterations=2)
        torch.testing.assert_close(result.adversarial, self.image)

    def test_invalid_parameters_and_prompts(self):
        for options in [dict(epsilon=-1), dict(epsilon=float("nan")), dict(iterations=0),
                        dict(iterations=1.5), dict(attack="unknown"), dict(diversity_probability=2)]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                attack_sam(self.adapter, self.image, **options)
        for options in [dict(), dict(point_coords=[[20, 4]]),
                        dict(point_coords=[[5, 4]], point_labels=[0.5]), dict(box=[5, 5, 1, 1])]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                SamLogits(ToySam(), (10, 20), **options)

    def test_cli_saved_pixels_metrics_and_transfer_are_consistent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path, checkpoint = root / "input.png", root / "sam.pth"
            pixels = (self.image[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            Image.fromarray(pixels).save(image_path)
            checkpoint.touch()
            args = get_argparser().parse_args([
                "--image", str(image_path), "--checkpoint", str(checkpoint),
                "--point", "5", "4", "1", "--device", "cpu", "--output", str(root / "output"),
                "--target", f"vit_l={checkpoint}",
            ])
            with patch("sam_attack.load_model", side_effect=lambda *args: ToySam()):
                report = run(args)
            stored = json.loads((args.output / "metrics.json").read_text())
            self.assertEqual(stored, report)
            attacked = np.array(Image.open(args.output / "adversarial.png"))
            delta = int(np.abs(attacked.astype(np.int16) - pixels.astype(np.int16)).max())
            self.assertLessEqual(delta, 12)
            self.assertEqual(delta, report["max_saved_pixel_delta"])
            self.assertGreater(report["source"]["adversarial_bce"], report["source"]["clean_bce"])
            self.assertEqual(report["targets"][0]["model"], "vit_l")
            self.assertTrue((args.output / "target_0_vit_l_adversarial_mask.png").is_file())

    def test_official_sam_modules_preserve_input_gradients_and_prediction(self):
        from segment_anything.modeling import Sam, ImageEncoderViT, PromptEncoder, MaskDecoder, TwoWayTransformer
        torch.manual_seed(3)
        sam = Sam(
            image_encoder=ImageEncoderViT(img_size=64, patch_size=16, in_chans=3, embed_dim=32,
                                         depth=1, num_heads=4, mlp_ratio=2, out_chans=32),
            prompt_encoder=PromptEncoder(embed_dim=32, image_embedding_size=(4, 4),
                                         input_image_size=(64, 64), mask_in_chans=8),
            mask_decoder=MaskDecoder(transformer_dim=32, transformer=TwoWayTransformer(
                depth=1, embedding_dim=32, num_heads=4, mlp_dim=64),
                iou_head_depth=2, iou_head_hidden_dim=32),
        )
        adapter = SamLogits(sam, (23, 31), [[12, 8]], [1])
        image = torch.rand(1, 3, 23, 31, requires_grad=True)
        logits = adapter(image)
        gradient, = torch.autograd.grad(logits.square().mean(), image)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0)
        resized = F.interpolate(image.detach(), (47, 64), mode="bilinear", align_corners=False, antialias=True)
        official = sam([{"image": resized[0] * 255, "original_size": (23, 31),
                         "point_coords": torch.tensor([[[12 * 64 / 31, 8 * 47 / 23]]]),
                         "point_labels": torch.tensor([[1]])}], multimask_output=False)[0]
        self.assertTrue(torch.equal(logits > 0, official["masks"]))
        result = attack_sam(adapter, image, attack="ni_di_ti", epsilon=12 / 255,
                            iterations=2, diversity_probability=1.0)
        self.assertEqual(result.adversarial.shape, image.shape)
        self.assertLessEqual((result.adversarial - image.detach()).abs().max().item(), 12 / 255 + 1e-7)
        self.assertGreater((result.adversarial - image.detach()).abs().max().item(), 0)
        self.assertTrue(all(parameter.grad is None for parameter in sam.parameters()))


if __name__ == "__main__":
    unittest.main()
