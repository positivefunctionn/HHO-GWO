# Fixed release patch notes

## Changes made

1. Added `experiments/evaluate_prediction_handoff.py` to evaluate the supplied precomputed BiLSTM and persistence predictions on a timestamp-sorted chronological tail segment. The JSON output explicitly states that this is a diagnostic, not a validated unseen-model test, because upstream prediction provenance is missing.
2. Added `results/prediction_handoff_diagnostic.json`, generated from the supplied 1,094-row handoff file using the final 20% (219 rows) and the declared overload threshold `actual_cpu >= 80`.
3. Updated `README.md` with the predictor provenance gap, correct scope of the synthetic scheduler benchmark, reproducible diagnostic command, and claim boundaries for proxy metrics and the HHO-GWO-specific selection policy.
4. Added regression tests for the new diagnostic metric functions.

## Verification

- Unit tests: 14 passed.
- Existing synthetic benchmark result files were preserved and not rerun or rewritten.
- Original handoff CSV was not modified.

## Remaining blockers that cannot honestly be patched from this archive alone

- Upstream BiLSTM training/inference code, model checkpoint, scaler, sequence construction, forecast horizon, train/validation/test split, and original training data are absent. The existing predictions cannot be certified as leakage-free or reproducible; rebuilding a new model would be a new experiment, not a repair of the original model.
- The supplied prediction diagnostic shows persistence outperforming BiLSTM on the final 20% segment (BiLSTM CPU MAE 3.883, RMSE 10.259, R² 0.264; persistence MAE 2.582, RMSE 8.736, R² 0.466). Only 2 of 219 tail rows meet the actual CPU overload threshold; therefore overload-classification statistics on this segment are unstable. These are diagnostic figures only and must not be reported as validated model test metrics.
- The scheduling benchmarks remain synthetic and do not validate end-to-end learned prediction, real cloud energy, or CloudSim performance. These require new evidence/data, not wording changes.
