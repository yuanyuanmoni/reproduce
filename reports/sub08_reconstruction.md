# Sub-08 Image Reconstruction Evidence

Status: **VERIFIED**. The fail-closed verifier accepted the real-data,
prediction-conditioned 200-image sub-08 run on 2026-09-29. The immutable
evidence manifest is `runs/sub08_reconstruction_seed0/reconstruction_evidence.json`.

The formal gate is:

```bash
python fuxian/scripts/verify_reconstruction_run.py \
  --run-dir fuxian/runs/sub08_reconstruction_seed0 \
  --data-root data_frame/Things_dataset/Things_eeg --report
```

Only a `VERIFIED` report from that command is a formal result. It requires
the actual official sub-08 200-image order, raw-80 EEG aggregation before ATM,
the prediction bundle as the sole generation condition, hash-bound training
targets/checkpoints/predictions/manifest/metrics, and exactly these five
metrics recomputed from the saved per-image CSV.

| Metric | Paper TS-Diff target | Observed status | Direction |
| --- | ---: | --- | --- |
| PixCorr | 0.176 | 0.073786 | higher |
| SSIM | 0.366 | 0.360123 | higher |
| Inception | 0.737 | 0.491256 | higher |
| CLIP | 0.799 | 0.501231 | higher |
| SwAV | 0.552 | 0.637070 | lower |

PixCorr, SSIM, Inception, CLIP and SwAV follow the local captureframe-compatible
definitions recorded in `metrics.json`. The paper does not fully specify all
evaluator implementation details; this run uses SDXL-Turbo at 512 px with a
ViT-H IP-Adapter, four deterministic steps and seed 0. It is therefore a
verified implementation-level reproduction, **not** a claim of exact paper
metric parity. The remaining gap is largest for semantic identification
(Inception/CLIP) and reflects the retrieval-to-prior-to-diffusion stack as
well as those implementation choices.
