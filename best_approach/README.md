# V60 Grand Master Pipeline

This directory contains the production-grade pipeline for the Bengawan Solo water level forecasting task: **V60 Grand Master**.

---

## Files

| File | Description |
|---|---|
| [`V60_Grand_Master.ipynb`](./V60_Grand_Master.ipynb) | End-to-end 14-stage execution notebook (data audit, feature engineering, robust Z-score transformation, 5-head heterogeneous ensemble, rolling-origin cross-validation, and submission generation). |
| [`create_nb_v60_grand_master.py`](./create_nb_v60_grand_master.py) | Programmatic Python script used to generate and maintain `V60_Grand_Master.ipynb`. |

---

## Execution Modes (`MODE`)

Cell 1 of [`V60_Grand_Master.ipynb`](./V60_Grand_Master.ipynb) exposes runtime flags to control execution scope:

```python
# Available mode options: "audit", "validate", "ablation", "tune", "fit", "all"
MODE = "all"

ENABLE_QUANTILE_MODEL = True   # Fits quantile regressors for uncertainty bounds (q10, q50, q90)
USE_SPIKE_WEIGHTS = True       # Applies reduced sample weights (0.05) to isolated sensor anomalies
PRUNE_FEATURES = False         # Prunes low-importance features based on ablation metrics
```

### Mode Descriptions:
1. **`"validate"`**: Executes 4-fold rolling-origin cross-validation without temporal data leakage. Reports Global RMSE, Macro-station MAE, and Worst-station MAE.
2. **`"fit"`**: Fits the 5-head ensemble on the entire dataset and generates `submission_V60_GrandMaster.csv`.
3. **`"tune"`**: Runs Optuna hyperparameter optimization specifically on the final temporal validation fold (`jan_2025`).
4. **`"ablation"`**: Evaluates performance differentials when removing spike weighting or specific feature subsets.
5. **`"all"`**: Runs the complete sequence: Data Audit $\to$ Validation $\to$ Final Fit $\to$ Quantile Forecasting.

---

## Ensemble Architecture

The model combines five diverse gradient boosted trees:

```python
DEFAULT_ENSEMBLE = (
    EnsembleMemberConfig("cb_shallow", "catboost", seed=17, depth=6, learning_rate=0.045, l2_leaf_reg=8.0),
    EnsembleMemberConfig("cb_base",    "catboost", seed=41, depth=8, learning_rate=0.02,  l2_leaf_reg=10.0),
    EnsembleMemberConfig("cb_deep",    "catboost", seed=83, depth=8, learning_rate=0.045, l2_leaf_reg=1.0),
    EnsembleMemberConfig("lgbm_a",     "lightgbm", seed=7,  num_leaves=63,  learning_rate=0.05, feature_fraction=0.85, bagging_fraction=0.8),
    EnsembleMemberConfig("lgbm_b",     "lightgbm", seed=29, num_leaves=127, learning_rate=0.05, feature_fraction=0.7,  bagging_fraction=0.8)
)
```

For complete theoretical derivations, mathematical formulations, and background on prior experimental iterations (V1 through V58), refer to the root [README](../README.md).
