"""Evaluate immutable reconstruction manifests with local captureframe-compatible metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from generate_reconstructions import CONDITION_SOURCE, assert_non_oracle_condition

PIXCORR_SIZE = 425
SSIM_SIZE = 425
PAPER_TARGETS = {"pixcorr": 0.176, "ssim": 0.366, "inception": 0.737, "clip": 0.799, "swav": 0.552}
METRIC_METADATA = {
    "pixcorr": {"direction": "higher", "definition": "RGB flattened Pearson r after bilinear resize to 425x425"},
    "ssim": {"direction": "higher", "definition": "grayscale SSIM at 425x425; Gaussian weights, sigma=1.5, population covariance"},
    "inception": {"direction": "higher", "definition": "InceptionV3 avgpool feature two-way cosine identification"},
    "clip": {"direction": "higher", "definition": "OpenAI CLIP ViT-L/14 feature Pearson two-way identification"},
    "swav": {"direction": "lower", "definition": "paired SwAV ResNet-50 avgpool Pearson correlation distance (1-r)"},
}


def load_rgb(path: str | Path, size: int | None = None) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return image.resize((size, size), Image.BILINEAR) if size is not None else image


def pixcorr(original: Image.Image, generated: Image.Image) -> float:
    a = np.asarray(original.resize((PIXCORR_SIZE, PIXCORR_SIZE), Image.BILINEAR), dtype=np.float32).reshape(-1)
    b = np.asarray(generated.resize((PIXCORR_SIZE, PIXCORR_SIZE), Image.BILINEAR), dtype=np.float32).reshape(-1)
    return 0.0 if np.std(a) == 0 or np.std(b) == 0 else float(np.corrcoef(a, b)[0, 1])


def ssim(original: Image.Image, generated: Image.Image) -> float:
    a = rgb2gray(np.asarray(original.resize((SSIM_SIZE, SSIM_SIZE), Image.BILINEAR), dtype=np.float32) / 255.0)
    b = rgb2gray(np.asarray(generated.resize((SSIM_SIZE, SSIM_SIZE), Image.BILINEAR), dtype=np.float32) / 255.0)
    return float(structural_similarity(b, a, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0))


def two_way_cosine(original: torch.Tensor, generated: torch.Tensor) -> np.ndarray:
    if original.shape != generated.shape or len(original) < 2:
        raise ValueError("Two-way identification requires matched feature matrices with at least two rows")
    sim = F.normalize(generated.float(), dim=1) @ F.normalize(original.float(), dim=1).T
    own = sim.diag().unsqueeze(1)
    wins = (own > sim)[~torch.eye(len(sim), dtype=torch.bool, device=sim.device)].view(len(sim), len(sim) - 1)
    return wins.float().mean(dim=1).cpu().numpy()


def paired_correlation_distance(original: torch.Tensor, generated: torch.Tensor) -> np.ndarray:
    if original.shape != generated.shape:
        raise ValueError("Paired correlation distance requires matched feature matrices")
    a = F.normalize(original.float() - original.float().mean(dim=1, keepdim=True), dim=1)
    b = F.normalize(generated.float() - generated.float().mean(dim=1, keepdim=True), dim=1)
    return (1.0 - (a * b).sum(dim=1)).cpu().numpy()


def pearson_two_way(original: torch.Tensor, generated: torch.Tensor) -> np.ndarray:
    if original.shape != generated.shape or len(original) < 2:
        raise ValueError("Pearson two-way identification requires matched matrices with at least two rows")
    a = F.normalize(original.float() - original.float().mean(dim=1, keepdim=True), dim=1)
    b = F.normalize(generated.float() - generated.float().mean(dim=1, keepdim=True), dim=1)
    correlation = a @ b.T
    own = correlation.diag().unsqueeze(0)
    wins = correlation < own
    wins.fill_diagonal_(False)
    return (wins.float().sum(dim=0) / (len(a) - 1)).cpu().numpy()


def validate_manifest(records: list[dict[str, Any]]) -> None:
    if len(records) < 2:
        raise ValueError("Evaluation manifest requires at least two records")
    expected_indices = list(range(len(records)))
    if [record.get("index") for record in records] != expected_indices:
        raise ValueError("Generation manifest index/order is invalid")
    identities = [(record.get("name"), record.get("image_id")) for record in records]
    if len(set(identities)) != len(records):
        raise ValueError("Generation manifest has duplicate identities")
    identity_hashes = {record.get("prediction_identity_hash") for record in records}
    if len(identity_hashes) != 1 or None in identity_hashes:
        raise ValueError("Generation manifest prediction identity hash is inconsistent")
    for record in records:
        assert_non_oracle_condition(record)
        for key in ("reference_path", "generated_path"):
            if not Path(record.get(key, "")).is_file():
                raise FileNotFoundError(f"Manifest {key} is missing: {record.get(key)}")


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    value = value.get("state_dict", value) if isinstance(value, dict) else value
    if not isinstance(value, dict):
        raise ValueError(f"Unsupported local metric checkpoint: {path}")
    return {str(key).removeprefix("module."): tensor for key, tensor in value.items()}


class FeatureMetric:
    def __init__(self, model: torch.nn.Module, layer: str, size: int, device: str):
        self.model, self.size, self.device, self.feature = model.eval().to(device), size, device, None
        modules = dict(self.model.named_modules())
        if layer not in modules:
            raise ValueError(f"Metric layer missing: {layer}")
        self.handle = modules[layer].register_forward_hook(lambda _m, _i, out: setattr(self, "feature", out.detach().flatten(1)))

    def encode_many(self, images: list[Image.Image]) -> torch.Tensor:
        values = []
        for image in images:
            array = np.asarray(image.resize((self.size, self.size), Image.BILINEAR), dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1)
            tensor = (tensor - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor([0.229, 0.224, 0.225])[:, None, None]
            self.feature = None
            with torch.no_grad(): self.model(tensor.unsqueeze(0).to(self.device))
            if self.feature is None: raise RuntimeError("Metric forward hook did not fire")
            values.append(self.feature.cpu())
        return torch.cat(values)

    def close(self) -> None: self.handle.remove()


class OpenAIClipMetric:
    def __init__(self, weights: Path, device: str):
        if not weights.is_file(): raise FileNotFoundError(f"OpenAI CLIP weights missing: {weights}")
        try: import clip
        except ImportError as error: raise RuntimeError("OpenAI CLIP package is required") from error
        self.model, _ = clip.load(str(weights), device=device, jit=False)
        self.device = device

    def encode_many(self, images: list[Image.Image]) -> torch.Tensor:
        values = []
        for image in images:
            array = np.asarray(image.resize((224, 224), Image.BILINEAR), dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1)
            tensor = (tensor - torch.tensor([0.48145466, 0.4578275, 0.40821073])[:, None, None]) / torch.tensor([0.26862954, 0.26130258, 0.27577711])[:, None, None]
            with torch.no_grad(): values.append(self.model.encode_image(tensor.unsqueeze(0).to(self.device)).float().cpu())
        return torch.cat(values)


def summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in PAPER_TARGETS}


def make_grid(records: list[dict[str, Any]], output: Path, limit: int = 100) -> None:
    rows = records[:limit]; size, label_h = 160, 24
    grid = Image.new("RGB", (size * 2, max(1, len(rows) * (size + label_h))), "white"); draw = ImageDraw.Draw(grid)
    for index, record in enumerate(rows):
        y = index * (size + label_h); grid.paste(load_rgb(record["reference_path"], size), (0, y + label_h)); grid.paste(load_rgb(record["generated_path"], size), (size, y + label_h))
        draw.text((2, y + 4), str(record["name"]), fill="black")
    grid.save(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inception-weights", type=Path, required=True)
    parser.add_argument("--openai-clip-weights", type=Path, required=True)
    parser.add_argument("--swav-weights", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not args.manifest.is_file(): raise FileNotFoundError(f"Manifest not found: {args.manifest}")
    for path in (args.inception_weights, args.openai_clip_weights, args.swav_weights):
        if not path.is_file(): raise FileNotFoundError(f"Metric weights must be local files: {path}")
    records = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    validate_manifest(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("metrics.csv", "metrics.json", "comparison_grid.jpg"):
        if (args.output_dir / name).exists(): raise FileExistsError(f"Refusing to overwrite immutable evaluation artifact: {name}")
    try: from torchvision import models
    except ImportError as error: raise RuntimeError("Evaluation requires torchvision") from error
    inception = models.inception_v3(weights=None, aux_logits=False, init_weights=False); inception.load_state_dict(_load_state_dict(args.inception_weights), strict=False)
    swav = models.resnet50(weights=None); swav.fc = torch.nn.Identity(); swav.load_state_dict(_load_state_dict(args.swav_weights), strict=False)
    inception_metric, swav_metric, clip_metric = FeatureMetric(inception, "avgpool", 342, args.device), FeatureMetric(swav, "avgpool", 224, args.device), OpenAIClipMetric(args.openai_clip_weights, args.device)
    originals, generated = [load_rgb(record["reference_path"]) for record in records], [load_rgb(record["generated_path"]) for record in records]
    inception_values = two_way_cosine(inception_metric.encode_many(originals), inception_metric.encode_many(generated)); swav_values = paired_correlation_distance(swav_metric.encode_many(originals), swav_metric.encode_many(generated)); clip_values = pearson_two_way(clip_metric.encode_many(originals), clip_metric.encode_many(generated))
    inception_metric.close(); swav_metric.close()
    rows = []
    for record, inc, sw, cl, original, image in zip(records, inception_values, swav_values, clip_values, originals, generated):
        rows.append({"index": record["index"], "name": record["name"], "image_id": record["image_id"], "pixcorr": pixcorr(original, image), "ssim": ssim(original, image), "inception": float(inc), "clip": float(cl), "swav": float(sw)})
    with (args.output_dir / "metrics.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    payload = {"paper_ts_diff_targets": PAPER_TARGETS, "metric_metadata": METRIC_METADATA, "summary": summary(rows), "count": len(rows), "manifest": str(args.manifest.resolve()), "condition_source": CONDITION_SOURCE}
    (args.output_dir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    make_grid(records, args.output_dir / "comparison_grid.jpg")
    print(json.dumps(payload["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
