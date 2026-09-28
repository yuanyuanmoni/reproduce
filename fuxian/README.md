# THINGS-EEG2 sub-08 NVOL retrieval reproduction

Retrieval only: this implementation contains no diffusion or image-generation components. It uses the fixed 200 held-out stimulus images and treats each ordered EEG/image pair as the sole correct answer.

Prepare the verified data contract and frozen visual caches (the cache tool never writes into `--data-root`):

```bash
python fuxian/tools/inspect_sub08_data.py --data-root data_frame/Things_dataset/Things_eeg --subject 8 --output fuxian/reports/data_contract.json
python fuxian/tools/cache_nvol_features.py --data-root data_frame/Things_dataset/Things_eeg --checkpoint /path/to/vit-h-14.bin --output-dir fuxian/cache --subject 8
```

Train only on the training split; `final_epoch.pt` is used without test-set selection:

```bash
python fuxian/train.py --data-root data_frame/Things_dataset/Things_eeg --visual-cache fuxian/cache/nvol_train_features.pt --output-dir fuxian/runs/sub08 --subject 8
```

Evaluate the complete ordered 200-image bank. The evaluator averages the 80 **raw EEG trials** for each test image before ATM, then L2-normalizes query embeddings. It verifies the subject, final-epoch checkpoint metadata, train-cache provenance, and test cache name/canonical-ID order before producing raw cosine and full-matrix CSLS results.

```bash
python fuxian/evaluate.py --data-root data_frame/Things_dataset/Things_eeg --checkpoint fuxian/runs/sub08/final_epoch.pt --train-visual-cache fuxian/cache/nvol_train_features.pt --test-visual-cache fuxian/cache/nvol_test_features.pt --output-dir fuxian/runs/sub08/evaluation --subject 8
```

The output contains `metrics.json`, full `scores.pt`, complete `rankings.pt`, and ordered query/candidate identities in `query_names.json`.
