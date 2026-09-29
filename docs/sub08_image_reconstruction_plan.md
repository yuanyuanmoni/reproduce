# Sub-08 Image Reconstruction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use task-by-task implementation and an independent review after every task.

**Goal:** Reproduce the paper's test-only, subject-08 two-stage image reconstruction path and report its five Table-III metrics with complete provenance.

**Architecture:** Freeze the existing final ATM checkpoint and frozen ViT-H/14 image encoder. Train a conditional DDPM from ATM embeddings to layer-12 NVOL targets, then train an NVOL-to-final-semantic adapter. At inference, aggregate all 80 raw test trials before ATM, generate a final feature through the two learned stages, and condition frozen local SDXL/IP-Adapter without any test-image feature. Reuse the local captureframe metric definitions for PixCorr, SSIM, Inception, CLIP, and SwAV, while recording that their evaluator settings supplement the paper.

**Tech Stack:** PyTorch, OpenCLIP ViT-H/14, local diffusers SDXL/IP-Adapter assets, Pillow, scikit-image, torchvision.

**Spec:** `基于脑电图的视觉检索与重建：从神经可感知的最优层到分层扩散生成.pdf`, Table III and Method III-C/D.

## Global Constraints

- Only train on the 1,654 training images and their four EEG trials; never use held-out test images, features, labels, or generated images for model selection.
- Test inference must average each image's 80 raw EEG trials before ATM and must use no test-stimulus visual feature as a generation condition.
- Use only local checkpoints; never download weights implicitly.
- Persist ordered image IDs, source/checkpoint SHA256 values, seed, model settings, generation manifest, per-image metrics, and summary metrics.
- Compare against paper TS-Diff targets: PixCorr .176, SSIM .366, Inception .737, CLIP .799, SwAV .552. Mark captureframe metric settings as implementation-defined where the paper is silent.

## Review Focus

- Test leakage: assert test visual features cannot enter either training command.
- Identity/order: assert cache, EEG, predicted feature, manifest, and metric rows share identical ordered 200 canonical image IDs.
- Aggregation: assert the test path averages raw `[200,80,63,250]` EEG before ATM exactly once.
- Condition source: assert each SDXL condition is derived from the saved predicted feature, never `original_path` pixels or oracle features.
- Metric direction/definition: assert SwAV is a distance (lower is better), while the other four match the captureframe evaluator definitions.

### Task 1: Provenanced reconstruction targets and protocol

**Files:**
- Create: `fuxian/src/nvol_retrieval/reconstruction_data.py`
- Create: `fuxian/tools/cache_reconstruction_targets.py`
- Test: `fuxian/tests/test_reconstruction_data.py`

**Interfaces:**
- Consumes: `FeatureCache`, official `load_sub08_split`, canonical image IDs, local ViT-H checkpoint.
- Produces: `ReconstructionTargetCache` containing ordered raw layer-12 `[N,1280]` targets and final semantic `[N,1024]` targets plus SHA256 provenance.

- [ ] Write tests for cache order rejection, duplicate IDs, and correct 80-trial raw mean.
- [ ] Implement a typed cache payload and local-only target-extraction command.
- [ ] Run the focused tests and inspect a two-image CPU smoke cache.
- [ ] Independent reviewer verifies cache provenance and leakage guards.

### Task 2: Conditional prior and semantic adapter

**Files:**
- Create: `fuxian/src/nvol_retrieval/reconstruction.py`
- Test: `fuxian/tests/test_reconstruction.py`

**Interfaces:**
- Consumes: frozen `ATMAlignmentModel` output `[B,1024]`, raw layer-12 target `[B,1280]` projected by the trained frozen retrieval layer-12 projection to `[B,1024]`, and final target `[B,1024]`.
- Produces: `ConditionalNVOLPrior`, `SemanticAdapter`, `prior_loss`, and `adapter_loss`.

- [ ] Write shape, deterministic sampling, loss-finiteness, and frozen-parameter tests.
- [ ] Implement the minimal paper-aligned DDPM loss terms and adapter losses.
- [ ] Run focused tests on CPU.
- [ ] Independent reviewer checks formula/gradient boundaries against Method III-C.

### Task 3: Train and export the two-stage decoder

**Files:**
- Create: `fuxian/train_reconstruction.py`
- Create: `fuxian/export_reconstruction_features.py`
- Test: `fuxian/tests/test_reconstruction_cli.py`

**Interfaces:**
- Consumes: final retrieval checkpoint and Task-1 training cache.
- Produces: final prior/adapter checkpoint and a 200-row test prediction bundle.

- [ ] Write CLI tests that reject test cache at training and reject incompatible ATM/checkpoint provenance.
- [ ] Implement final-epoch training (no validation selection), 80-trial aggregation export, metadata, and hashes.
- [ ] Run one-batch CPU smoke and the focused tests.
- [ ] Independent reviewer checks that every test feature is post-training and that final epoch is used.

### Task 4: Local SDXL/IP-Adapter generation and metric evaluation

**Files:**
- Create: `fuxian/generate_reconstructions.py`
- Create: `fuxian/evaluate_reconstructions.py`
- Test: `fuxian/tests/test_reconstruction_evaluation.py`

**Interfaces:**
- Consumes: Task-3 prediction bundle and explicitly supplied local SDXL, IP-Adapter, and metric-weight paths.
- Produces: generated PNGs, immutable `manifest.jsonl`, `metrics.csv`, `metrics.json`, and comparison grid.

- [ ] Write tests for manifest identity checks, condition-source guards, PixCorr/SSIM values, and metric summary directions.
- [ ] Implement local-only generation with a fixed seed and metric reuse compatible with captureframe's Table-III definitions.
- [ ] Run a two-image generation smoke only after local asset preflight passes; then run full 200-image evaluation as a separate explicit job.
- [ ] Independent reviewer verifies the manifest and all five metric definitions/outputs.

### Task 5: Formal evidence report

**Files:**
- Create: `fuxian/scripts/verify_reconstruction_run.py`
- Create: `fuxian/reports/sub08_reconstruction.md`
- Test: `fuxian/tests/test_reconstruction_verifier.py`

**Interfaces:**
- Consumes: Tasks 1–4 artifacts.
- Produces: pass/fail evidence verdict and a paper-versus-observed metric table.

- [ ] Write verifier tests for missing files, altered image order, wrong aggregation label, and non-local/oracle condition provenance.
- [ ] Implement fail-closed verification and the report with the paper values and evaluator caveat.
- [ ] Run all reconstruction tests and independently recompute the saved metric summaries.
- [ ] Independent reviewer signs or blocks the evidence bundle.

## Self-Review

Coverage: tasks 1–5 respectively cover targets/provenance, the two learned stages, train/test separation, synthesis/metrics, and evidence. The five stated review risks each have a dedicated test in the owning task. The paper omits DDPM schedule, U-Net width, SDXL sampling parameters, and exact metric feature layers; those must be saved in run metadata and described as implementation-defined rather than fabricated as paper facts.
