#!/usr/bin/env bash
set -euo pipefail

SEED=42
FOLDS=artifacts/folds/entrywise_5_folds_seed_${SEED}.npz

python -m scripts.create_folds \
  --input data/ratings.npy \
  --output "${FOLDS}" \
  --n_splits 5 \
  --seed ${SEED}

python -m scripts.prepare_features \
  --genres_path data/genres.npy \
  --years_path data/years.npy \
  --genres_norm_method row_l1 \
  --genres_impute_method none \
  --years_norm_method col_zscore \
  --years_impute_method col_median \
  --output_path data/features.npz

python -m scripts.tune_params \
  --R_path data/ratings.npy \
  --folds_path "${FOLDS}" \
  --features_path data/features.npz \
  --study_name als_tune_v1 \
  --n_trials 150 \
  --seed ${SEED} \
  --save_every 25 \
  --verbose_fit 0

python -m scripts.evaluate_models \
  --R_path data/ratings.npy \
  --folds_path "${FOLDS}" \
  --best_params_path results/tuning/als_tune_v1_best_params.json \
  --features_path data/features.npz \
  --out_dir results/ablations \
  --n_pop_bins 5 \
  --es_tol 1e-4 \
  --es_min_iters 10 \
  --verbose_fit 0
