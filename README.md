# Bengawan Solo Flood Forecasting: Multi-Horizon Hydrological Intelligence

[![Architecture](https://img.shields.io/badge/Architecture-Target--Normalized%20Global%20Ensemble-blue.svg?style=flat-square)](#recommended-architecture-v60-grand-master)
[![Validation](https://img.shields.io/badge/Validation-Rolling--Origin%204--Fold-green.svg?style=flat-square)](#validation-strategy-rolling-origin)
[![Metric](https://img.shields.io/badge/Evaluation%20Metric-RMSE%20%7C%20MAE-orange.svg?style=flat-square)](#problem-formulation-and-evaluation-metric)
[![Status](https://img.shields.io/badge/Project%20Status-Complete-success.svg?style=flat-square)](#execution-guide)

This repository contains the research, hydrological domain modeling, and machine learning pipeline developed for multi-horizon water level forecasting (*Tinggi Muka Air* / TMA) across 30 monitoring stations along the Bengawan Solo River Basin in Indonesia.

---

## Table of Contents
1. [Problem Formulation and Evaluation Metric](#problem-formulation-and-evaluation-metric)
2. [Repository Structure](#repository-structure)
3. [Hydrological Challenges and Data Constraints](#hydrological-challenges-and-data-constraints)
4. [Experimental Evolution: From V1 to V60](#experimental-evolution-from-v1-to-v60)
5. [Recommended Architecture: V60 Grand Master](#recommended-architecture-v60-grand-master)
6. [Mathematical Formulations](#mathematical-formulations)
7. [Validation Strategy: Rolling-Origin](#validation-strategy-rolling-origin)
8. [Interactive Spatial Map](#interactive-spatial-map)
9. [Execution Guide](#execution-guide)

---

## Problem Formulation and Evaluation Metric

### Task Definition
The objective is to predict water surface elevation (`tma_mdpl`, measured in meters above mean sea level) across **30 hydrological monitoring stations** for the deployment period from **September 19, 2025 to May 18, 2026** (242 consecutive days, or approximately 8 months).

* **Observation Timestamps:** 06:00, 12:00, and 18:00 Western Indonesia Time (WIB) daily.
* **Forecast Horizon:** 242 days $\times$ 3 timestamps/day $\times$ 30 stations = **21,780 target predictions**.
* **Primary Evaluation Metric:** Root Mean Squared Error (RMSE):

$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N \left( y_i - \hat{y}_i \right)^2}$$

---

## Repository Structure

The workspace is organized into modular directories:

```text
SSF-SSDS/
├── best_approach/                        # Production-ready implementation
│   ├── README.md                         # Quick-start instructions for V60
│   ├── V60_Grand_Master.ipynb            # End-to-end 14-stage execution notebook
│   └── create_nb_v60_grand_master.py     # Script to generate V60 notebook
│
├── data_pendukung/                       # Environmental and spatial data
│   ├── data_lingkungan.csv               # Hourly weather variables (Open-Meteo, 2023-2026)
│   ├── koordinat_pos.csv                 # Geographic coordinates of all 30 stations
│   └── HydroRIVERS_v10_au_shp/           # River reach topology shapefiles
│
├── docs/                                 # Technical documentation and system designs
│   ├── architecture.md                   # Complete architectural specification
│   └── V36_Architecture_Design.md        # Transition analysis from local to global models
│
├── experiments/                          # Historical model development records
│   ├── notebooks/                        # 51 notebooks covering versions V1 through V58
│   └── generators/                       # 50 Python generator scripts for reproducible builds
│
├── scripts/                              # Geospatial utilities and diagnostic tools
│   ├── create_river_topology_map.py      # Script to render the interactive Folium map
│   ├── extract_true_topology.py          # Traversal tool for HydroRIVERS reach graphs
│   ├── deep_scan.py                      # Data validation and missing-value auditor
│   └── hydro_mapping.json                # Mapping of river reaches to station identifiers
│
├── visualizations/                       # Rendered visualization artifacts
│   └── Peta_Aliran_Bengawan_Solo.html    # Interactive map with dynamic stream animation
│
├── train.csv                             # Historical observed water level records
├── test.csv                              # Target timestamps and station IDs for inference
├── sample_submission.csv                 # Target submission format
├── .gitignore                            # Excludes large raw datasets (>100MB) from git
└── README.md                             # Comprehensive technical documentation
```

---

## Hydrological Challenges and Data Constraints

### 1. Spatial Elevation Heterogeneity
The Bengawan Solo river basin exhibits extreme differences in base elevation:
* Upstream stations (e.g., Waduk Wonogiri) operate near **140 meters** above sea level.
* Downstream stations (e.g., Babat, Bojonegoro) operate at elevations between **1 and 10 meters** above sea level.

A model trained on raw elevation in meters tends to minimize squared errors on high-elevation stations while neglecting downstream flood stages.

### 2. Operational 8-Month Horizon Without Target Lags
During the test period, true water levels are unavailable. Models that rely on recursive autoregressive lags ($y_{t-1}, y_{t-2}$) compound prediction errors over 242 days, leading to divergence when seasonal weather regimes transition. Consequently, the primary pipeline strictly uses **pure exogenous meteorological forcing**.

### 3. Sensor Glitches and Sentinel Values
* Historical sensor records contain single-timestamp non-physical jumps (e.g., Napel station spiking momentarily from $34.95\text{ m} \to 325.83\text{ m} \to 34.55\text{ m}$). Rather than dropping rows, these points are damped using sample weighting ($w_i = 0.05$) combined with Huber loss.
* Future weather projections include `-999` sentinels for solar radiation, which are replaced by missing value indicators and forward-filled causally (maximum 24 hours for atmospheric variables, 31 days for slow-moving macro-climate indices such as Nino 3.4).

---

## Experimental Evolution: From V1 to V60

Model development progressed through distinct phases to address specific failure modes:

```text
[Phase 1: Deep Learning (V1 - V12)]
  - Evaluated: LSTM, Spatial-Temporal GCN (ST-GCN), Graph WaveNet (GWNet).
  - Outcome: Underperformed due to limited sample length (2.7 years) and distribution shift over 8 months.
       |
       v
[Phase 2: Tree-Based Exogenous Models (V13 - V25)]
  - Evaluated: XGBoost, LightGBM, and CatBoost with weather lags and hydrological soil indices.
  - Outcome: Significant performance gains; gradient boosting handled tabular meteorological features robustly.
       |
       v
[Phase 3: The Local Models Trap (V26 & V35)]
  - Evaluated: 30 independent local models (one per station).
  - V26 achieved a strong Cross-Validation score (0.98) using Tweedie loss on raw targets.
  - V35 expanded V26 with 84 complex features ("The Local GOAT").
  - Outcome: V35 suffered severe overfitting on the public leaderboard (LB score degraded to 1.93)
             because 84 features overwhelmed the small local sample size (~900 observations per station).
       |
       v
[Phase 4: Transition to Global Modeling (V36 - V58)]
  - Evaluated: Single global models trained across all 27,000 composite observations.
  - Outcome: Global models prevented overfitting (LB 1.60), but training on raw meters suppressed
             localized flood peaks due to variance compression across heterogeneous elevations.
       |
       v
[Phase 5: The Winning Synthesis (V60 Grand Master)]
  - Evaluated: Station-specific Robust Z-Score Normalization + Global 5-Head Ensemble + Huber Loss.
  - Outcome: Successfully resolved both the local overfitting and global variance compression issues.
```

---

## Recommended Architecture: V60 Grand Master

The final production model in [`best_approach/V60_Grand_Master.ipynb`](./best_approach/V60_Grand_Master.ipynb) integrates the key findings into an end-to-end framework:

### Core Pillars
1. **Station-Wise Robust Z-Score Normalization:** Standardizes target water levels per station so all 30 stations share a common reference scale. This enables a single global model to learn runoff dynamics without station-elevation bias.
2. **Sensor Spike Suppression via Huber Loss:** Minimizes sensitivity to sensor anomalies while maintaining gradient response during severe flood events.
3. **5-Head Heterogeneous Ensemble:** Combines three CatBoost configurations and two LightGBM configurations with varying depths, regularization rates, and random seeds.
4. **Hydrological Feature Formulation:** Integrates Antecedent Precipitation Index (API), effective runoff estimates, and spatial upstream precipitation routing derived from HydroRIVERS stream network topology.
5. **Multi-Horizon Rolling-Origin Validation:** Evaluates performance across four historical 8-month deployment windows to ensure temporal generalization without data leakage.

---

## Mathematical Formulations

### 1. Robust Z-Score Target Normalization
For station $p$ at observation time $t$:

$$Z(p, t) = \frac{\text{TMA}(p, t) - \text{Median}_{\text{train}}(p)}{\max\left(\frac{\text{IQR}_{\text{train}}(p)}{1.349}, \ 0.05\right)}$$

Where:
* $\text{Median}_{\text{train}}(p)$ is the median water level for station $p$ computed on the training fold.
* $\text{IQR}_{\text{train}}(p) = Q_3(p) - Q_1(p)$ is the interquartile range of station $p$.
* The factor $1.349$ normalizes the IQR to standard deviation under a Gaussian assumption.
* The floor $\max(\cdot, 0.05)$ prevents division by zero for stations with minimal water level variance.

### 2. Inverse Transformation for Inference
Predictions $\widehat{Z}(p, t)$ produced by the global ensemble are mapped back to physical elevation (meters):

$$\widehat{\text{TMA}}(p, t) = \left( \widehat{Z}(p, t) \times \max\left(\frac{\text{IQR}_{\text{train}}(p)}{1.349}, \ 0.05\right) \right) + \text{Median}_{\text{train}}(p)$$

### 3. Huber Loss Objective
To limit the influence of sensor noise without discarding physical flood peaks:

$$L_\delta(z, \hat{z}) = \begin{cases} 
\frac{1}{2}(z - \hat{z})^2 & \text{if } |z - \hat{z}| \le \delta \\
\delta |z - \hat{z}| - \frac{1}{2}\delta^2 & \text{if } |z - \hat{z}| > \delta
\end{cases}$$

Here, $\delta = 1.5$. Isolated sensor spikes receive a reduced sample weight $w_i = 0.05$, whereas continuous flood events retain standard weighting $w_i = 1.0$.

### 4. Antecedent Precipitation Index (API)
Soil moisture memory and saturation are modeled using Exponential Weighted Moving averages of precipitation $P$:

$$\text{API}_t = \sum_{k=0}^{\infty} \alpha (1 - \alpha)^k P_{t-k}, \quad \alpha = 1 - \exp\left(-\frac{\ln 2}{\tau_{\text{half}}}\right)$$

Three distinct hydrological half-lives are computed:
* Fast response (surface runoff): $\tau_{\text{half}} = 12\text{ hours}$
* Medium response (root zone saturation): $\tau_{\text{half}} = 48\text{ hours}$
* Slow response (baseflow recharge): $\tau_{\text{half}} = 168\text{ hours (7 days)}$

### 5. Effective Runoff and Upstream Routing
Immediate direct runoff is approximated by interacting current precipitation with shallow soil moisture:

$$\text{Runoff}_t = P_t \times \bar{\theta}_{t, 0-28\text{cm}}$$

Upstream meteorological signals are routed to downstream stations through hydraulic distance decay based on HydroRIVERS network reach distances:

$$W(u \to v) = \exp\left(-\frac{d_{\text{hydraulic}}(u, v)}{75\text{ km}}\right)$$

---

## Validation Strategy: Rolling-Origin

To simulate the operational deployment environment and prevent forward-looking data leakage, validation is structured across four temporal cutoffs:

| Fold Identifier | Training Cutoff Date | Validation Period | Validation Horizon |
|:---:|:---:|:---:|:---:|
| `sep_2023` | Up to Sep 18, 2023 | Sep 19, 2023 – May 18, 2024 | 242 days (8 months) |
| `may_2024` | Up to May 18, 2024 | May 19, 2024 – Jan 18, 2025 | 242 days (8 months) |
| `sep_2024` | Up to Sep 18, 2024 | Sep 19, 2024 – May 18, 2025 | 242 days (8 months) |
| `jan_2025` | Up to Jan 18, 2025 | Jan 19, 2025 – Sep 18, 2025 | 242 days (8 months) |

Evaluation tracks Global RMSE, Mean Absolute Error (MAE), Macro-station MAE, and Worst-station MAE.

---

## Interactive Spatial Map

An interactive geographical representation of the river network is available in [`visualizations/Peta_Aliran_Bengawan_Solo.html`](./visualizations/Peta_Aliran_Bengawan_Solo.html).

Key features:
* Rendered on CartoDB Positron tiles using exact station coordinates.
* Color-coded elevation gradient indicating upstream (high elevation) to downstream (low elevation).
* Animated stream paths (*AntPath*) illustrating downstream hydraulic connectivity.

To inspect the map locally:
```powershell
Start-Process .\visualizations\Peta_Aliran_Bengawan_Solo.html
```

---

## Execution Guide

### 1. Environment Setup
Install dependencies:
```bash
pip install -q pandas numpy scikit-learn catboost lightgbm xgboost optuna folium nbformat
```

### 2. Running the V60 Pipeline
Open [`best_approach/V60_Grand_Master.ipynb`](./best_approach/V60_Grand_Master.ipynb) in JupyterLab, VS Code, Google Colab, or Kaggle.

Select the operational mode in **Cell 1**:
```python
# Available modes: "fit", "validate", "tune", "ablation", "all"
MODE = "fit"                 # Trains final 5-head ensemble on full data and outputs submission.csv
ENABLE_QUANTILE_MODEL = True # Generates lower (q10), median (q50), and upper (q90) flood boundaries
USE_SPIKE_WEIGHTS = True     # Applies sample weight damping (0.05) to isolated sensor jumps
```

Data paths automatically resolve whether the notebook is executed from the repository root, from within the `best_approach/` directory, or on Google Colab.
