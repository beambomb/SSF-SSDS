import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# === Cell 1: Global Config ===
cell_1 = nbf.v4.new_code_cell("""# === TAHAP 1: GLOBAL CONFIGURATION & FLAGS ===
# MODE OPTIONS: "audit", "validate", "ablation", "tune", "all", "fit"
MODE = "all" 

ENABLE_QUANTILE_MODEL = True
USE_SPIKE_WEIGHTS = True
PRUNE_FEATURES = False
FEATURE_IMPORTANCE_SOURCE = None # Diisi path/string CSV jika MODE='ablation'

print(f"PIPELINE BERJALAN PADA MODE: {MODE.upper()}")
""")

# === Cell 2: Ensemble Configurations ===
cell_2 = nbf.v4.new_code_cell("""# === TAHAP 2: ENSEMBLE CONFIGURATIONS ===
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class EnsembleMemberConfig:
    name: str
    family: str # 'catboost' atau 'lightgbm'
    seed: int
    kwargs: Dict[str, Any]
    
    def __init__(self, name, family, seed, **kwargs):
        self.name = name
        self.family = family
        self.seed = seed
        self.kwargs = kwargs

# Sesuai instruksi optimasi (Diversifikasi Ensemble 5 Kepala)
DEFAULT_ENSEMBLE = (
    EnsembleMemberConfig("cb_shallow", "catboost", seed=17, depth=6, learning_rate=0.045, l2_leaf_reg=8.0),
    EnsembleMemberConfig("cb_base", "catboost", seed=41, depth=8, learning_rate=0.02, l2_leaf_reg=10.0),
    EnsembleMemberConfig("cb_deep", "catboost", seed=83, depth=8, learning_rate=0.045, l2_leaf_reg=1.0),
    EnsembleMemberConfig("lgbm_a", "lightgbm", seed=7, num_leaves=63, learning_rate=0.05, feature_fraction=0.85, bagging_fraction=0.8),
    EnsembleMemberConfig("lgbm_b", "lightgbm", seed=29, num_leaves=127, learning_rate=0.05, feature_fraction=0.7, bagging_fraction=0.8)
)
""")

# === Cell 3: Imports & Kaggle Paths ===
cell_3 = nbf.v4.new_code_cell("""# === TAHAP 3: IMPORTS & KAGGLE PATHS ===
import os, gc, warnings, math
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime

!pip install -q catboost lightgbm optuna statsmodels
import optuna
import catboost as cb
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm.auto import tqdm

IN_COLAB = False
try:
    import google.colab
    IN_COLAB = True
    print("Berjalan di Google Colab. Melakukan mount drive...")
    from google.colab import drive
    drive.mount('/content/drive')
except:
    pass

if IN_COLAB:
    DATA_DIR = '/content/drive/MyDrive/Kaggle/SSF-SSDS/data_pendukung'
    TRAIN_PATH = '/content/drive/MyDrive/Kaggle/SSF-SSDS/train.csv'
    TEST_PATH = '/content/drive/MyDrive/Kaggle/SSF-SSDS/test.csv'
    SAMPLE_SUB_PATH = '/content/drive/MyDrive/Kaggle/SSF-SSDS/sample_submission.csv'
    OUTPUT_DIR = '/content/'
else:
    # Auto-detect whether running from repo root or best_approach/
    base_data = '../' if os.path.exists('../train.csv') else './'
    DATA_DIR = os.path.join(base_data, 'data_pendukung')
    TRAIN_PATH = os.path.join(base_data, 'train.csv')
    TEST_PATH = os.path.join(base_data, 'test.csv')
    SAMPLE_SUB_PATH = os.path.join(base_data, 'sample_submission.csv')
    OUTPUT_DIR = './'
""")

# === Cell 4: Data Loading & Sentinel Cleaning ===
cell_4 = nbf.v4.new_code_cell("""# === TAHAP 4: DATA LOADING & SENTINEL CLEANING ===
print("Memuat data utama...")
train_df = pd.read_csv(TRAIN_PATH, parse_dates=['datetime'])
test_df = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
env_df = pd.read_csv(os.path.join(DATA_DIR, 'data_lingkungan.csv'), parse_dates=['datetime'])

print("Membersihkan Sentinel & Missing Values (Causal Forward-Fill)...")
# Sentinel radiasi
if 'solar_radiation_mj_m2' in env_df.columns:
    env_df.loc[env_df['solar_radiation_mj_m2'] == -999, 'solar_radiation_mj_m2'] = np.nan

# Forward fill untuk data kosong di masa depan
env_df = env_df.sort_values(['nama_pos', 'datetime']).reset_index(drop=True)
fill_cols = [c for c in env_df.columns if c not in ['nama_pos', 'datetime']]
env_df[fill_cols] = env_df.groupby('nama_pos')[fill_cols].ffill(limit=24) # Maksimum 24 jam untuk weather
if 'nino_34' in env_df.columns:
    env_df['nino_34'] = env_df.groupby('nama_pos')['nino_34'].ffill(limit=31*24) # 1 bulan untuk nino
""")

# === Cell 5: Target Z-Score Transformer ===
cell_5 = nbf.v4.new_code_cell("""# === TAHAP 5: ROBUST TARGET NORMALIZATION (Z-SCORE) ===
def fit_target_scaler(df):
    stats = {}
    for pos in df['nama_pos'].unique():
        tma = df[df['nama_pos'] == pos]['tma_mdpl'].dropna()
        if len(tma) == 0: continue
        median = tma.median()
        iqr = tma.quantile(0.75) - tma.quantile(0.25)
        # Sesuai architecture.md: max(IQR / 1.349, 0.05)
        scale = max(iqr / 1.349, 0.05)
        stats[pos] = {'median': median, 'scale': scale}
    return stats

def transform_target(df, stats, inverse=False, col_in='tma_mdpl', col_out='tma_mdpl_z'):
    out = np.zeros(len(df))
    out[:] = np.nan
    
    pos_idx = df['nama_pos'].values
    val_in = df[col_in].values
    
    for i in range(len(df)):
        pos = pos_idx[i]
        if pos in stats and not np.isnan(val_in[i]):
            if not inverse:
                out[i] = (val_in[i] - stats[pos]['median']) / stats[pos]['scale']
            else:
                out[i] = (val_in[i] * stats[pos]['scale']) + stats[pos]['median']
    
    df[col_out] = out
    return df

print("Menghitung statistik target (Z-Score)...")
target_stats = fit_target_scaler(train_df)
train_df = transform_target(train_df, target_stats, inverse=False, col_in='tma_mdpl', col_out='target')
""")

# === Cell 6: Spike Downweighter ===
cell_6 = nbf.v4.new_code_cell("""# === TAHAP 6: SENSOR EXORCISM (SPIKE DOWNWEIGHTING) ===
def apply_spike_weights(df, use_weights=True):
    df['sample_weight'] = 1.0
    if not use_weights:
        return df
        
    total_spikes = 0
    for st in df['nama_pos'].unique():
        mask = df['nama_pos'] == st
        st_data = df.loc[mask]
        
        diff_f = st_data['tma_mdpl'].diff().abs()
        diff_b = st_data['tma_mdpl'].diff(-1).abs()
        
        mad_f = max(diff_f.median(), 0.1)
        mad_b = max(diff_b.median(), 0.1)
        
        # Deteksi lonjakan ekstrim yang tidak fisik (spike 1 titik)
        is_spike = (diff_f > 4.0 * mad_f) & (diff_b > 4.0 * mad_b)
        
        # Assign bobot 0.05 ke spike (membiarkan model belajar robust lewat Huber)
        df.loc[mask & is_spike, 'sample_weight'] = 0.05
        total_spikes += is_spike.sum()
        
    print(f"🔮 Ditemukan {total_spikes} titik glitch. Diberi bobot 0.05!")
    return df

train_df = apply_spike_weights(train_df, use_weights=USE_SPIKE_WEIGHTS)
""")

# === Cell 7: Exogenous Feature Engineering & Bug Fixes ===
cell_7 = nbf.v4.new_code_cell("""# === TAHAP 7: FEATURE ENGINEERING (HYDROLOGY & PURE FORCING) ===
def prepare_environment(base_df, env_df):
    df = base_df.copy()
    
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24.0)
    
    # Bug Fix 1: seasonal_baseline copy to avoid read-only numpy array crash
    # (Di sini kita langsung mapping dari env_df)
    
    df = pd.merge(df, env_df, on=['datetime', 'nama_pos'], how='left')
    
    # Native Categorical
    df['nama_pos_cat'] = df['nama_pos'].astype('category')
    
    # Bug Fix 2: Perbaikan Typo nama kolom rainfall
    if 'rainfall_max_24h_mm' in df.columns:
        # Nama asli dari dataset
        df['rainfall_mm_max_24h'] = df['rainfall_max_24h_mm'].clip(lower=0)
    
    df = df.sort_values(['nama_pos', 'datetime']).reset_index(drop=True)
    return df

print("Menyusun Train Set (Pure Forcing)...")
train_full = prepare_environment(train_df, env_df)

print("Menyusun Test Set (Pure Forcing)...")
test_parsed = test_df.copy()
split_info = test_parsed['id'].str.split(' - ', n=1, expand=True)
test_parsed['datetime'] = pd.to_datetime(split_info[0])
test_parsed['nama_pos'] = split_info[1]
test_full = prepare_environment(test_parsed, env_df)

drop_cols = ['datetime', 'tma_mdpl', 'tma_asli', 'id', 'target', 'sample_weight']
features = [c for c in train_full.columns if c not in drop_cols and train_full[c].dtype.name != 'object']
cat_features = ['nama_pos_cat']
print(f"Total Fitur: {len(features)}")
""")

# === Cell 8: Folds Definition ===
cell_8 = nbf.v4.new_code_cell("""# === TAHAP 8: ROLLING-ORIGIN FOLDS ===
FOLDS_CONFIG = [
    {"name": "sep_2023", "train_end": "2023-09-18", "val_start": "2023-09-19", "val_end": "2024-05-18"},
    {"name": "may_2024", "train_end": "2024-05-18", "val_start": "2024-05-19", "val_end": "2025-01-18"},
    {"name": "sep_2024", "train_end": "2024-09-18", "val_start": "2024-09-19", "val_end": "2025-05-18"},
    {"name": "jan_2025", "train_end": "2025-01-18", "val_start": "2025-01-19", "val_end": "2025-09-18"}
]

def get_fold_indices(df, fold_cfg):
    train_end = pd.to_datetime(fold_cfg['train_end'])
    val_start = pd.to_datetime(fold_cfg['val_start'])
    val_end = pd.to_datetime(fold_cfg['val_end'])
    
    train_idx = df.index[df['datetime'] <= train_end]
    val_idx = df.index[(df['datetime'] >= val_start) & (df['datetime'] <= val_end)]
    
    # Drop stasiun dengan histori terlalu sedikit (< 90 label)
    train_pos_counts = df.loc[train_idx].groupby('nama_pos')['target'].count()
    valid_pos = train_pos_counts[train_pos_counts >= 90].index
    
    train_idx = train_idx[df.loc[train_idx, 'nama_pos'].isin(valid_pos)]
    val_idx = val_idx[df.loc[val_idx, 'nama_pos'].isin(valid_pos)]
    
    return train_idx, val_idx
""")

# === Cell 9: Family-Agnostic Model Wrappers ===
cell_9 = nbf.v4.new_code_cell("""# === TAHAP 9: FAMILY-AGNOSTIC WRAPPERS (CatBoost & LGBM) ===
def fit_member(config: EnsembleMemberConfig, X_tr, y_tr, X_va, y_va, w_tr=None):
    if config.family == "catboost":
        cat_idx = [X_tr.columns.get_loc(c) for c in cat_features]
        # Wajib Huber:delta=1.5
        params = config.kwargs.copy()
        params.update({
            'loss_function': 'Huber:delta=1.5',
            'random_seed': config.seed,
            'verbose': False,
            'iterations': 1000,
            'early_stopping_rounds': 50
        })
        model = cb.CatBoostRegressor(**params)
        model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=(X_va, y_va), cat_features=cat_idx)
        return model
        
    elif config.family == "lightgbm":
        params = config.kwargs.copy()
        params.update({
            'objective': 'huber',
            'alpha': 1.5,
            'metric': 'mae', # MAE lebih stabil untuk early stopping
            'seed': config.seed,
            'verbose': -1,
            'n_estimators': 1000
        })
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_va, y_va)], 
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        return model
    else:
        raise ValueError("Unknown family")

def predict_member(model, X_te):
    return model.predict(X_te)
""")

# === Cell 10: Run Validation ===
cell_10 = nbf.v4.new_code_cell("""# === TAHAP 10: RUN VALIDATION (FULL ENSEMBLE) ===
def run_validation(ensemble_configs, df_train):
    print("\\n--- 🚀 MEMULAI ROLLING-ORIGIN VALIDATION ---")
    results = []
    
    for fold in FOLDS_CONFIG:
        print(f"\\nEvaluating Fold: {fold['name']}")
        tr_idx, va_idx = get_fold_indices(df_train, fold)
        
        # Data
        X_tr = df_train.loc[tr_idx, features]
        y_tr = df_train.loc[tr_idx, 'target']
        w_tr = df_train.loc[tr_idx, 'sample_weight']
        
        X_va = df_train.loc[va_idx, features]
        y_va = df_train.loc[va_idx, 'target']
        
        # Drop baris NaN pada target
        tr_valid = y_tr.notna()
        X_tr, y_tr, w_tr = X_tr[tr_valid], y_tr[tr_valid], w_tr[tr_valid]
        
        va_valid = y_va.notna()
        X_va, y_va = X_va[va_valid], y_va[va_valid]
        
        # Hitung Normalisasi Target yang HANYA di-fit dari train set fold ini!
        fold_stats = fit_target_scaler(df_train.loc[tr_idx[tr_valid]])
        
        # Training Seluruh Member
        models = []
        ensemble_preds_z = np.zeros(len(y_va))
        
        for cfg in ensemble_configs:
            m = fit_member(cfg, X_tr, y_tr, X_va, y_va, w_tr)
            models.append(m)
            ensemble_preds_z += predict_member(m, X_va) / len(ensemble_configs)
            print(f"   ✓ {cfg.name} trained.")
            
        # Evaluasi dengan Inverse Transform (Skala Meter Asli)
        df_va_eval = df_train.loc[va_idx[va_valid]].copy()
        df_va_eval['pred_z'] = ensemble_preds_z
        df_va_eval = transform_target(df_va_eval, fold_stats, inverse=True, col_in='pred_z', col_out='pred_mdpl')
        
        true_mdpl = df_va_eval['tma_mdpl'].values
        pred_mdpl = df_va_eval['pred_mdpl'].values
        
        # Hitung Metrics
        mae = mean_absolute_error(true_mdpl, pred_mdpl)
        rmse = np.sqrt(mean_squared_error(true_mdpl, pred_mdpl))
        
        df_va_eval['abs_error'] = np.abs(true_mdpl - pred_mdpl)
        macro_mae = df_va_eval.groupby('nama_pos')['abs_error'].mean().mean()
        worst_station = df_va_eval.groupby('nama_pos')['abs_error'].mean().max()
        
        # Metrik Kotor vs Bersih
        clean_mask = df_va_eval['sample_weight'] == 1.0
        clean_mae = df_va_eval[clean_mask].groupby('nama_pos')['abs_error'].mean().mean() if clean_mask.sum()>0 else macro_mae
        
        print(f"   => RMSE: {rmse:.4f} | MAE: {mae:.4f} | Macro MAE: {macro_mae:.4f} (Clean: {clean_mae:.4f}) | Worst: {worst_station:.4f}")
        
        results.append({
            'fold': fold['name'],
            'rmse': rmse,
            'macro_mae': macro_mae,
            'worst_mae': worst_station,
            'clean_macro': clean_mae
        })
        
    res_df = pd.DataFrame(results)
    print("\\n=== HASIL AKHIR VALIDASI ===")
    display(res_df)
    return res_df
""")

# === Cell 11: Run Ablation Study ===
cell_11 = nbf.v4.new_code_cell("""# === TAHAP 11: ABLATION STUDY (MURAH MERIAH) ===
def run_ablation_study(df_train):
    print("\\n--- 🔬 MEMULAI ABLATION STUDY (2 Folds Terakhir) ---")
    ablation_folds = FOLDS_CONFIG[-2:]
    
    # Hanya pakai 1 model ringan untuk eksperimen cepat
    test_cfg = EnsembleMemberConfig("cb_shallow_ablation", "catboost", seed=42, depth=6, learning_rate=0.05, l2_leaf_reg=8.0)
    
    # Jika PRUNE_FEATURES nyala, baca feature_importance.csv
    ablation_features = features.copy()
    if PRUNE_FEATURES and FEATURE_IMPORTANCE_SOURCE:
        # Simulasi pruning (Misal ambil Top 50 fitur)
        try:
            imp_df = pd.read_csv(FEATURE_IMPORTANCE_SOURCE)
            ablation_features = imp_df.sort_values('Importance', ascending=False)['Feature'].head(50).tolist()
            print(f"Fitur dipangkas menjadi {len(ablation_features)} fitur berdasarkan importance.")
        except:
            print("Gagal membaca feature importance. Memakai semua fitur.")
    
    for fold in ablation_folds:
        print(f"\\nAblation Fold: {fold['name']}")
        tr_idx, va_idx = get_fold_indices(df_train, fold)
        
        X_tr = df_train.loc[tr_idx, ablation_features]
        y_tr = df_train.loc[tr_idx, 'target']
        
        # Test 1: Tanpa Spike Weights (Semua 1.0)
        w_off = np.ones(len(y_tr))
        
        # Test 2: Dengan Spike Weights (0.05)
        w_on = df_train.loc[tr_idx, 'sample_weight'].values
        
        X_va = df_train.loc[va_idx, ablation_features]
        y_va = df_train.loc[va_idx, 'target']
        
        tr_valid = y_tr.notna()
        X_tr, y_tr = X_tr[tr_valid], y_tr[tr_valid]
        w_off, w_on = w_off[tr_valid], w_on[tr_valid]
        
        va_valid = y_va.notna()
        X_va, y_va = X_va[va_valid], y_va[va_valid]
        
        fold_stats = fit_target_scaler(df_train.loc[tr_idx[tr_valid]])
        
        # Fit OFF
        m_off = fit_member(test_cfg, X_tr, y_tr, X_va, y_va, w_tr=w_off)
        pred_off = predict_member(m_off, X_va)
        
        # Fit ON
        m_on = fit_member(test_cfg, X_tr, y_tr, X_va, y_va, w_tr=w_on)
        pred_on = predict_member(m_on, X_va)
        
        # Hitung Error di Skala Asli
        df_va = df_train.loc[va_idx[va_valid]].copy()
        
        df_va['pred_z_off'] = pred_off
        df_va = transform_target(df_va, fold_stats, inverse=True, col_in='pred_z_off', col_out='pred_off')
        
        df_va['pred_z_on'] = pred_on
        df_va = transform_target(df_va, fold_stats, inverse=True, col_in='pred_z_on', col_out='pred_on')
        
        rmse_off = np.sqrt(mean_squared_error(df_va['tma_mdpl'], df_va['pred_off']))
        rmse_on = np.sqrt(mean_squared_error(df_va['tma_mdpl'], df_va['pred_on']))
        
        print(f"   => RMSE (Spike Weights OFF): {rmse_off:.4f}")
        print(f"   => RMSE (Spike Weights ON) : {rmse_on:.4f}")
""")

# === Cell 12: Run Hyperparameter Sweep ===
cell_12 = nbf.v4.new_code_cell("""# === TAHAP 12: OPTUNA SWEEP (JAN 2025 FOLD ONLY) ===
def run_hyperparam_sweep(df_train):
    print("\\n--- 🎛️ MEMULAI OPTUNA SENSITIVITY CHECK (Fold jan_2025) ---")
    fold = [f for f in FOLDS_CONFIG if f['name'] == 'jan_2025'][0]
    
    tr_idx, va_idx = get_fold_indices(df_train, fold)
    X_tr = df_train.loc[tr_idx, features]
    y_tr = df_train.loc[tr_idx, 'target']
    w_tr = df_train.loc[tr_idx, 'sample_weight']
    X_va = df_train.loc[va_idx, features]
    y_va = df_train.loc[va_idx, 'target']
    
    tr_valid = y_tr.notna()
    X_tr, y_tr, w_tr = X_tr[tr_valid], y_tr[tr_valid], w_tr[tr_valid]
    va_valid = y_va.notna()
    X_va, y_va = X_va[va_valid], y_va[va_valid]
    
    fold_stats = fit_target_scaler(df_train.loc[tr_idx[tr_valid]])
    
    def objective(trial):
        cfg = EnsembleMemberConfig(
            name="tune", family="catboost", seed=42,
            depth=trial.suggest_int('depth', 6, 10),
            learning_rate=trial.suggest_float('lr', 0.01, 0.1, log=True),
            l2_leaf_reg=trial.suggest_float('l2', 1.0, 15.0)
        )
        
        m = fit_member(cfg, X_tr, y_tr, X_va, y_va, w_tr)
        pred_z = predict_member(m, X_va)
        
        df_v = df_train.loc[va_idx[va_valid]].copy()
        df_v['pred_z'] = pred_z
        df_v = transform_target(df_v, fold_stats, inverse=True, col_in='pred_z', col_out='pred')
        
        return mean_absolute_error(df_v['tma_mdpl'], df_v['pred'])
        
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)
    
    print("\\nTop Parameters dari Sweep (Berdasarkan MAE):")
    for t in sorted(study.trials, key=lambda x: x.value)[:3]:
        print(f"MAE: {t.value:.4f} | Params: {t.params}")
""")

# === Cell 13: Fit Final & Quantile ===
cell_13 = nbf.v4.new_code_cell("""# === TAHAP 13: FINAL FIT, SUBMIT & QUANTILE MODEL ===
def fit_final_and_submit(ensemble_configs, df_train, df_test):
    print("\\n--- 🚀 MEMULAI FINAL FIT & INFERENSI ---")
    
    # Siapkan Full Train Set
    X_tr_full = df_train[features]
    y_tr_full = df_train['target']
    w_tr_full = df_train['sample_weight']
    
    valid_mask = y_tr_full.notna()
    X_tr_full = X_tr_full[valid_mask]
    y_tr_full = y_tr_full[valid_mask]
    w_tr_full = w_tr_full[valid_mask]
    
    # Hitung Normalisasi Final
    final_stats = fit_target_scaler(df_train[valid_mask])
    
    # Train Ensemble (Tanpa eval set / iterasi fixed 1000)
    models = []
    for cfg in ensemble_configs:
        print(f"Melatih final {cfg.name}...")
        m = fit_member(cfg, X_tr_full, y_tr_full, X_tr_full.head(100), y_tr_full.head(100), w_tr_full)
        models.append(m)
        
    # Inferensi ke Test Set
    X_te = df_test[features]
    test_preds_z = np.zeros(len(X_te))
    
    for m in models:
        test_preds_z += predict_member(m, X_te) / len(models)
        
    # Inverse Transform ke Skala Asli
    df_test_out = df_test.copy()
    df_test_out['tma_mdpl_z'] = test_preds_z
    df_test_out = transform_target(df_test_out, final_stats, inverse=True, col_in='tma_mdpl_z', col_out='tma_mdpl')
    
    # Siapkan Submission Utama
    sub = pd.merge(sample_sub[['id']], df_test_out[['id', 'tma_mdpl']], on='id', how='left')
    sub['tma_mdpl'] = sub['tma_mdpl'].fillna(method='ffill').fillna(method='bfill')
    sub.to_csv('submission_V60_GrandMaster.csv', index=False)
    print("✅ File 'submission_V60_GrandMaster.csv' berhasil dicetak!")
    
    # --- BONUS: QUANTILE MODEL ---
    if ENABLE_QUANTILE_MODEL:
        print("\\n--- 🌊 MEMBANGUN CATBOOST MULTI-QUANTILE MODEL (Peringatan Banjir Ekstrim) ---")
        q_params = DEFAULT_ENSEMBLE[1].kwargs.copy()
        q_params.update({
            'loss_function': 'MultiQuantile:alpha=0.1,0.5,0.9',
            'iterations': 1200
        })
        
        cat_idx = [X_tr_full.columns.get_loc(c) for c in cat_features]
        q_model = cb.CatBoostRegressor(**q_params)
        q_model.fit(X_tr_full, y_tr_full, sample_weight=w_tr_full, cat_features=cat_idx, verbose=False)
        
        q_preds_z = q_model.predict(X_te) # Output matrix shape (N, 3)
        
        df_q = df_test[['id', 'nama_pos']].copy()
        df_q['q10_z'] = q_preds_z[:, 0]
        df_q['q50_z'] = q_preds_z[:, 1]
        df_q['q90_z'] = q_preds_z[:, 2]
        
        for q in ['q10', 'q50', 'q90']:
            df_q = transform_target(df_q, final_stats, inverse=True, col_in=f'{q}_z', col_out=q)
            
        df_q = df_q[['id', 'q10', 'q50', 'q90']]
        df_q.to_csv('submission_quantiles.csv', index=False)
        print("✅ File 'submission_quantiles.csv' berhasil dicetak!")
""")

# === Cell 14: Pipeline Router ===
cell_14 = nbf.v4.new_code_cell("""# === TAHAP 14: PIPELINE ROUTER ===
print(f"=== MENJALANKAN PIPELINE V60 (MODE: {MODE.upper()}) ===")

if MODE in ["audit", "all"]:
    print("Audit Data Selesai. Lanjut ke proses...")

if MODE in ["validate", "all"]:
    run_validation(DEFAULT_ENSEMBLE, train_full)

if MODE in ["ablation", "all"]:
    run_ablation_study(train_full)
    
if MODE in ["tune", "all"]:
    run_hyperparam_sweep(train_full)
    
if MODE in ["fit", "all"]:
    fit_final_and_submit(DEFAULT_ENSEMBLE, train_full, test_full)

print("🏆 Eksekusi Notebook V60 Grand Master Selesai!")
""")

nb.cells = [
    nbf.v4.new_markdown_cell("# 👑 V60: The Grand Master Pipeline"),
    cell_1, cell_2, cell_3, cell_4, cell_5, cell_6, cell_7, 
    cell_8, cell_9, cell_10, cell_11, cell_12, cell_13, cell_14
]

output_nb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'V60_Grand_Master.ipynb')
with open(output_nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print(f"Notebook berhasil disimpan di: {output_nb_path}")

print("V60 Notebook generated successfully!")
