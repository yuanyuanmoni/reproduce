# sub-08 retrieval reproduction status

Status: formally executed and verified (2026-09-28).

The runnable protocol is `scripts/run_sub08_reproduction.sh`. It first checks
the required Python packages, writes an environment export and exact command
record outside the raw data root, then performs inspection, feature caching,
final-epoch-only training, evaluation, and fail-closed verification. Its
`--synthetic-smoke` mode writes only synthetic scores and metadata; it neither
reads nor writes raw EEG data. The verifier deliberately rejects that output as
formal evidence; the command reports only that this rejection behaved correctly.

The verifier requires the persisted aggregation label
`mean_80_raw_eeg_before_atm`: all 80 trials for each held-out image are averaged
before the ATM encoder, before embedding normalization and ranking.

The formal run used the fixed 200-image candidate bank, a train-only visual
feature cache, and final-epoch checkpoint evaluation. The independent
fail-closed verifier accepted the saved evidence bundle. Observed metrics were:

| metric | Top-1 | Top-5 |
| --- | ---: | ---: |
| Raw cosine | 78.5% | 98.0% |
| CSLS | 90.5% | 98.5% |

These results are close to the paper reference values of raw Top-1/Top-5
84.5/98.0 and CSLS Top-1/Top-5 91.0/99.5. The formal run's caches,
checkpoint, scores, rankings, and logs remain local under `runs/` and are
deliberately excluded from version control because they are derived artifacts
and include large files.
