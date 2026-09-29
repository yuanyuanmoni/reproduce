# THINGS-EEG2 sub-08 retrieval protocol

This directory reproduces the retrieval branch only. No diffusion, reconstruction,
SDXL, ControlNet, or IP-Adapter code is included. An official repository for the
target paper was not found in the 2026-09-24 search. `captureframe` is used only
as the user-approved local reference for THINGS-EEG2 preprocessing and ATM.

| Item | Target paper / requested protocol | `captureframe` reference | Current reproduction |
| --- | --- | --- | --- |
| Dataset | THINGS-EEG2, sub-08 | Same local sub-08 arrays | Official `(16540,4,63,250)` train and `(200,80,63,250)` test arrays |
| Preprocessing | EEG trial protocol | Four train trials flattened; test trials meaned before encoder | Same semantics; duplicate caption paths never create EEG trials |
| Repeats | Four train positives; 80 test trials | Same | Four independent train inputs, exact 80-trial test mean |
| ATM | Temporal attention EEG encoder | Cogcap ATM | One temporal TransformerEncoderLayer, sub-08 linear (250→250), `PatchEmbedding(40, channel_kernel=63)`, 36×40=1440 flatten, 1024-D projection, L2 normalization |
| Visual teacher | frozen OpenCLIP ViT-H/14 blocks 10/12/14 | Existing projects use OpenCLIP visual features, but not this fusion | Cache frozen mean-pooled `[N,3,1280]`; train independent 1280→1024 linear + LayerNorm, softmax fusion, L2 normalize |
| Loss | symmetric multi-positive InfoNCE | Contrastive alignment patterns | Task 3: same-image trials are positives |
| Candidate bank | Complete fixed 200 test stimuli | Test manifest | Ordered 200-image gallery; no sampling or class metric |
| Aggregation | Mean 80 EEG trials before ranking | Mean before EEG encoder | Enforced by evaluation task |
| CSLS | Full 200×200 score matrix | Not used as authority | Task 4 implementation |
| Metrics | Exact-image raw Top-1/Top-5 84.5/98.0; CSLS 91.0/99.5 | Not comparable | Report observed values without claiming reproduction unless achieved |

Visual backbone parameters are frozen and permanently remain in evaluation mode.
The persisted evaluation metadata is `trial_aggregation="mean_80_raw_eeg_before_atm"`:
all 80 held-out trials are averaged before ATM encoding, L2 normalization, and
ranking. The cache contains only frozen mean-pooled block features `[N,3,1280]`, not a
randomly initialized fused target. `NVOLLayerFusion` owns the three trainable
projection layers, LayerNorms, and fusion logits; Task 3 must instantiate it and
jointly optimize it with ATM. A cache contains `names`, canonical image paths
relative to `--data-root`, and a SHA-256 hash over both in order; both cache
integrity and requested order are verified on load.

The ATM path is transcribed locally from the approved Cogcap reference and excludes
all generation/reconstruction heads. Its resolved trainable parameter count is
2,970,201 (also recorded in training metadata); the `Proj_eeg` residual branch follows
the reference (Linear→GELU→Linear→Dropout plus residual and LayerNorm). ATM,
`NVOLLayerFusion`, and the learned log-temperature (initialized as `log(1/0.07)`) are
jointly optimized with AdamW (`3e-4`), batch size 100, for 30 epochs. The final epoch
is saved without looking at the 200-image test set. Training batches are shuffled by
image ID but preserve all four flattened EEG trials per image (25 complete image groups
at batch size 100); the final smaller complete-group batch is retained by default, or
`--incomplete-batch error` fails closed.

Extraction requires `open_clip_torch` (import name `open_clip`) exposing the
`create_model_and_transforms` API; the preflight logs its detected version, plus a local ViT-H/14
checkpoint. `cache_nvol_features.py` performs this preflight before loading the
checkpoint, rejects non-positive batch sizes, and refuses any output directory
inside `--data-root` to protect raw data.

## Test preflight

Tests require `pytest>=7` in addition to PyTorch and Pillow. Before running the
focused visual-teacher tests, verify the selected environment and then run:

```bash
conda run --no-capture-output -n env_pytorch python -c "import pytest; print(pytest.__version__)"
PYTHONPATH=fuxian/src conda run --no-capture-output -n env_pytorch python -m pytest fuxian/tests/test_vision.py -q
```

As checked on 2026-09-24, the supplied `env_pytorch` does **not** have pytest
installed, so this preflight fails with `ModuleNotFoundError`; no dependency was
installed by this reproduction work.
