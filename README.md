# THINGS-EEG2 sub-08 retrieval and image-reconstruction reproduction

This repository reproduces the subject-08 path in *EEG-based Visual Retrieval
and Reconstruction: From Neurally Visible Optimal Layer to Hierarchical
Diffusion Generation*. It contains both the NVOL retrieval branch and the
two-stage reconstruction branch:

1. ATM maps EEG to a retrieval embedding and aligns frozen ViT-H/14 NVOL features.
2. A conditional diffusion prior reconstructs layer-12 NVOL features, and a semantic adapter maps them to the final CLIP conditioning feature used by frozen local SDXL/IP-Adapter generation.

All commands below are relative to the repository root. Data, visual caches,
checkpoints, experiment logs and generated images are deliberately ignored by
Git; provide local paths for them when running the workflow.

## Retrieval

Prepare the verified split contract and frozen visual caches. The cache tools
never write into `--data-root`.

```bash
python tools/inspect_sub08_data.py --data-root /path/to/Things_eeg --subject 8 --output reports/data_contract.json
python tools/cache_nvol_features.py --data-root /path/to/Things_eeg --checkpoint /path/to/vit-h-14.bin --output-dir cache --subject 8
```

Train only on the training split, then evaluate the ordered 200-image test
bank. Evaluation averages the 80 raw test EEG trials exactly once before ATM.

```bash
python train.py --data-root /path/to/Things_eeg --visual-cache cache/nvol_train_features.pt --output-dir runs/sub08 --subject 8
python evaluate.py --data-root /path/to/Things_eeg --checkpoint runs/sub08/final_epoch.pt --train-visual-cache cache/nvol_train_features.pt --test-visual-cache cache/nvol_test_features.pt --output-dir runs/sub08/evaluation --subject 8
```

## Image reconstruction

The reconstruction plan and paper-aligned constraints are in
[`docs/sub08_image_reconstruction_plan.md`](docs/sub08_image_reconstruction_plan.md).
The scripts enforce train-only target caches, final-epoch checkpoint selection,
and a prediction-conditioned (non-oracle) test path:

```bash
python tools/cache_reconstruction_targets.py --help
python train_reconstruction.py --help
python export_reconstruction_features.py --help
python generate_reconstructions.py --help
python evaluate_reconstructions.py --help
```

For a completed run, the fail-closed verifier checks the official 200-image
order, raw-80 aggregation, checkpoint/cache hashes, prediction-only generation
conditions, and the persisted per-image metrics:

```bash
python scripts/verify_reconstruction_run.py --run-dir /path/to/run --data-root /path/to/Things_eeg --report
```

The formal sub-08 summary is in
[`reports/sub08_reconstruction.md`](reports/sub08_reconstruction.md). The
reported result is implementation-level evidence, not a claim of exact paper
metric parity.

## Tests

```bash
pytest -q tests
```
