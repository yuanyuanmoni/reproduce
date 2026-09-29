#!/usr/bin/env bash
# Run the formal sub-08 pipeline, or create only synthetic non-EEG evidence.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}; DATA_ROOT=; CHECKPOINT=; OUTPUT_DIR=; DEVICE=${DEVICE:-cuda}; SEED=${SEED:-0}; DISABLE_CUDNN=0; SMOKE=0
usage() { echo "usage: $0 --output-dir PATH [--synthetic-smoke] | --data-root PATH --checkpoint PATH --output-dir PATH [--device DEVICE] [--seed N] [--disable-cudnn]"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT=$2; shift 2 ;;
    --checkpoint) CHECKPOINT=$2; shift 2 ;;
    --output-dir) OUTPUT_DIR=$2; shift 2 ;;
    --device) DEVICE=$2; shift 2 ;;
    --seed) SEED=$2; shift 2 ;;
    --disable-cudnn) DISABLE_CUDNN=1; shift ;;
    --synthetic-smoke) SMOKE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
[[ -n "$OUTPUT_DIR" ]] || { usage >&2; exit 2; }
mkdir -p "$OUTPUT_DIR"
if [[ "$SMOKE" == 1 ]]; then
  "$PYTHON" - "$OUTPUT_DIR" <<'PY'
import hashlib, json, sys
from pathlib import Path
import torch
run = Path(sys.argv[1]); evaluation = run / "evaluation"; evaluation.mkdir(exist_ok=True)
names = [f"synthetic_{i:03d}" for i in range(200)]; ids = [f"image_set/synthetic/{i:03d}.jpg" for i in range(200)]
digest = hashlib.sha256(json.dumps({"names": names, "image_ids": ids}, separators=(",", ":")).encode()).hexdigest()
metadata = {"subject": 8, "checkpoint_selection": "final_epoch", "cache_hash": "synthetic-train-cache", "cache_file_sha256": "synthetic-train-file"}
(run / "train_metadata.json").write_text(json.dumps(metadata, indent=2))
record = {"subject": 8, "checkpoint_selection": "final_epoch", "trial_aggregation": "mean_80_raw_eeg_before_atm", "ground_truth": "exact_image_identity", "commands": ["synthetic-smoke"], "environment_export": "synthetic", "device": "cpu", "seed": 0, "train_cache_hash": "synthetic-train-cache", "test_cache_hash": "synthetic-test-cache", "train_cache_file_sha256": "synthetic-train-file", "metrics": {"raw_top1": 1.0, "raw_top5": 1.0, "csls_top1": 1.0, "csls_top5": 1.0}}
(run / "run_metadata.json").write_text(json.dumps(record, indent=2))
(evaluation / "query_names.json").write_text(json.dumps({"names": names, "image_ids": ids, "candidate_name_hash": digest}, indent=2))
(evaluation / "metrics.json").write_text(json.dumps({"candidate_name_hash": digest, "trial_aggregation": "mean_80_raw_eeg_before_atm", "raw_top1": 1.0, "raw_top5": 1.0, "csls_top1": 1.0, "csls_top5": 1.0}, indent=2))
scores = torch.eye(200); order = scores.argsort(1, descending=True); ranks = order.argsort(1).diagonal() + 1
torch.save({"metadata": metadata}, run / "final_epoch.pt")
torch.save({"raw_scores": scores, "csls_scores": scores}, evaluation / "scores.pt")
torch.save({"raw_ranks": ranks, "csls_ranks": ranks, "raw_order": order, "csls_order": order}, evaluation / "rankings.pt")
PY
  if "$PYTHON" "$ROOT/scripts/verify_run.py" --run-dir "$OUTPUT_DIR"; then
    echo "synthetic smoke unexpectedly passed formal verification" >&2
    exit 1
  fi
  echo "SMOKE PASS: synthetic artifacts were correctly rejected as formal evidence"
  exit 0
fi
[[ -n "$DATA_ROOT" && -n "$CHECKPOINT" ]] || { usage >&2; exit 2; }
CHECKPOINT=$("$PYTHON" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$CHECKPOINT")
"$PYTHON" - "$DATA_ROOT" "$OUTPUT_DIR" <<'PY'
import sys
from pathlib import Path
data, output = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
if output.is_relative_to(data): raise SystemExit("--output-dir must be outside --data-root to preserve raw data")
PY
"$PYTHON" -c 'import numpy, PIL, torch, open_clip; print("dependencies: torch/open_clip/numpy/Pillow")'
"$PYTHON" -c 'import pytest; print("pytest=" + pytest.__version__)'
if command -v conda >/dev/null; then conda env export > "$OUTPUT_DIR/environment.yml"; else "$PYTHON" -m pip freeze > "$OUTPUT_DIR/environment.yml"; fi
CACHE_CUDNN_ARG=()
if [[ "$DISABLE_CUDNN" == 1 ]]; then CACHE_CUDNN_ARG=(--disable-cudnn); fi
commands=(
  "$PYTHON $ROOT/tools/inspect_sub08_data.py --data-root $DATA_ROOT --subject 8 --output $OUTPUT_DIR/data_contract.json"
  "$PYTHON $ROOT/tools/cache_nvol_features.py --data-root $DATA_ROOT --checkpoint $CHECKPOINT --output-dir $OUTPUT_DIR/cache --subject 8 --device $DEVICE ${CACHE_CUDNN_ARG[*]}"
  "$PYTHON $ROOT/train.py --data-root $DATA_ROOT --visual-cache $OUTPUT_DIR/cache/nvol_train_features.pt --output-dir $OUTPUT_DIR --subject 8 --seed $SEED --device $DEVICE ${CACHE_CUDNN_ARG[*]}"
  "$PYTHON $ROOT/evaluate.py --data-root $DATA_ROOT --checkpoint $OUTPUT_DIR/final_epoch.pt --train-visual-cache $OUTPUT_DIR/cache/nvol_train_features.pt --test-visual-cache $OUTPUT_DIR/cache/nvol_test_features.pt --output-dir $OUTPUT_DIR/evaluation --subject 8 --device $DEVICE ${CACHE_CUDNN_ARG[*]}"
)
printf '%s\n' "${commands[@]}" > "$OUTPUT_DIR/commands.txt"
"$PYTHON" "$ROOT/tools/inspect_sub08_data.py" --data-root "$DATA_ROOT" --subject 8 --output "$OUTPUT_DIR/data_contract.json"
"$PYTHON" "$ROOT/tools/cache_nvol_features.py" --data-root "$DATA_ROOT" --checkpoint "$CHECKPOINT" --output-dir "$OUTPUT_DIR/cache" --subject 8 --device "$DEVICE" "${CACHE_CUDNN_ARG[@]}"
"$PYTHON" "$ROOT/train.py" --data-root "$DATA_ROOT" --visual-cache "$OUTPUT_DIR/cache/nvol_train_features.pt" --output-dir "$OUTPUT_DIR" --subject 8 --seed "$SEED" --device "$DEVICE" "${CACHE_CUDNN_ARG[@]}"
"$PYTHON" "$ROOT/evaluate.py" --data-root "$DATA_ROOT" --checkpoint "$OUTPUT_DIR/final_epoch.pt" --train-visual-cache "$OUTPUT_DIR/cache/nvol_train_features.pt" --test-visual-cache "$OUTPUT_DIR/cache/nvol_test_features.pt" --output-dir "$OUTPUT_DIR/evaluation" --subject 8 --device "$DEVICE" "${CACHE_CUDNN_ARG[@]}"
"$PYTHON" - "$OUTPUT_DIR" "$DEVICE" "$SEED" <<'PY'
import hashlib, json, sys
from pathlib import Path
import torch
run, device, seed = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):
    try: return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError: return torch.load(path, map_location="cpu")
checkpoint = load(run / "final_epoch.pt"); test = load(run / "cache/nvol_test_features.pt")
metrics = json.loads((run / "evaluation/metrics.json").read_text())
contract = json.loads((run / "data_contract.json").read_text())
provenance = {split: json.loads((run / "cache" / f"nvol_{split}_features.provenance.json").read_text()) for split in ("train", "test")}
record = {"real_data": True, "subject": 8, "checkpoint_selection": "final_epoch", "trial_aggregation": "mean_80_raw_eeg_before_atm", "ground_truth": "exact_image_identity", "commands": (run / "commands.txt").read_text().splitlines(), "environment_export": "environment.yml", "device": device, "seed": seed, "data_contract_sha256": sha(run / "data_contract.json"), "data_contract_ordered_names_sha256": {split: contract[split]["ordered_names_sha256"] for split in ("train", "test")}, "cache_identity_hash_kind": "canonical_name_and_image_id_sha256", "visual_checkpoint": provenance["train"]["source_checkpoint"], "train_cache_hash": checkpoint["metadata"]["cache_hash"], "test_cache_hash": test["name_hash"], "train_cache_file_sha256": sha(run / "cache/nvol_train_features.pt"), "test_cache_file_sha256": sha(run / "cache/nvol_test_features.pt"), "cache_provenance": provenance, "metrics": {key: metrics[key] for key in ("raw_top1", "raw_top5", "csls_top1", "csls_top5")}}
(run / "run_metadata.json").write_text(json.dumps(record, indent=2) + "\n")
PY
exec "$PYTHON" "$ROOT/scripts/verify_run.py" --run-dir "$OUTPUT_DIR"
