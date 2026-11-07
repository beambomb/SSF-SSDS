import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

def create_v30_notebook():
    nb = new_notebook()
    
    cells = [
        new_markdown_cell('# 🧟 V30 THE ULTIMATE FRANKENSTEIN (Dual-Engine) 🧟\\nMenjawab tantangan skor 1.61 dengan menggabungkan **SEMUA** eksperimen terhebat kita ke dalam 1 notebook tanpa kompromi!\\n- **Mesin Utama:** Dual-Engine per pos (XGBoost Tweedie + Random Forest).\\n- **Validasi:** Rolling Origin Temporal CV 3-Fold (Anti Data Leakage).\\n- **Fitur:** Semua Jangkar V26 + Fisika Air V27 (Dew Point/EWM) + Topologi Spasial Hulu-Hilir.\\n- **Filter:** Sensor Anomaly Downweighting (0.05).'),
        
        new_markdown_cell('## 1. Persiapan Environment & Data'),
        new_code_cell('''!pip install -q xgboost scikit-learn
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error

plt.style.use('seaborn-v0_8-whitegrid')

if not 'path' in locals():
    if os.path.exists('/kaggle/input'):
        dataset_folders = [f for f in os.listdir('/kaggle/input') if os.path.isdir(f'/kaggle/input/{f}')]
        path = f'/kaggle/input/{dataset_folders[0]}/' if dataset_folders else '/kaggle/input/'
    elif os.path.exists('/content'):
        from google.colab import drive
        drive.mount('/content/drive')
        path = '/content/drive/MyDrive/SSF-SSDS/'
    else:
        path = './'

train_df = pd.read_csv(path + 'train.csv', parse_dates=['datetime'])
test_df = pd.read_csv(path + 'test.csv')
lingkungan_df = pd.read_csv(path + 'data_pendukung/data_lingkungan.csv', parse_dates=['datetime'])
sample_sub = pd.read_csv(path + 'sample_submission.csv')
coords = pd.read_csv(path + 'data_pendukung/koordinat_pos.csv')

print("Data berhasil dimuat!")
'''),

        new_markdown_cell('## 2. Peta Topologi & Kompilasi Fitur "God-Tier"'),
        new_code_cell('''print("Membangun Peta Topologi Spasial Hulu-Hilir...")

elevations = train_df.groupby('nama_pos')['tma_mdpl'].mean().to_dict()
coords['elevation'] = coords['nama_pos'].map(elevations)
upstream_map = {pos: [] for pos in coords['nama_pos']}

for i, row_i in coords.iterrows():
    pos_i = row_i['nama_pos']
    elev_i = row_i['elevation']
    candidates = coords[coords['elevation'] < elev_i].copy()
    if len(candidates) > 0:
        dist = np.sqrt((candidates['latitude'] - row_i['latitude'])**2 + (candidates['longitude'] - row_i['longitude'])**2)
        nearest_idx = dist.idxmin()
        pos_hilir = coords.loc[nearest_idx, 'nama_pos']
        upstream_map[pos_hilir].append(pos_i)

print("Mengekstraksi Fitur Fisika Air (V27) & Hujan (V26)...")
lingkungan_df = lingkungan_df.sort_values(['nama_pos', 'datetime']).reset_index(drop=True)

def build_exogenous_features(env_df):
    agg_df = env_df[['datetime', 'nama_pos']].copy()
    grouped = env_df.groupby('nama_pos')
    
    # 📌 V27: DEW-POINT DEPRESSION (Evaporasi)
    if 'temperature_c' in env_df.columns and 'humidity_pct' in env_df.columns:
        env_df['dew_point'] = env_df['temperature_c'] - ((100 - env_df['humidity_pct']) / 5.0)
        env_df['dew_point_dep'] = env_df['temperature_c'] - env_df['dew_point']
        agg_df['dew_point_dep_mean_6h'] = grouped['dew_point_dep'].transform(lambda x: x.rolling(6, min_periods=1).mean())
        
    # 📌 V27: ANTECEDENT PRECIPITATION INDEX (EWM)
    if 'rainfall_mm' in env_df.columns:
        agg_df['api_48h'] = grouped['rainfall_mm'].transform(lambda x: x.ewm(halflife=48, min_periods=1).mean())
        agg_df['api_168h'] = grouped['rainfall_mm'].transform(lambda x: x.ewm(halflife=168, min_periods=1).mean())
        
    # 📌 V26: ROLLING SUM HUJAN KLASIK
    if 'rainfall_mm' in env_df.columns:
        agg_df['rain_sum_6h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(6, min_periods=1).sum())
        agg_df['rain_sum_72h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(72, min_periods=1).sum())
        agg_df['rain_sum_168h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(168, min_periods=1).sum())
        
    # Standard weather
    for c in ['temperature_c', 'wind_speed_kmh', 'surface_pressure_hpa', 'humidity_pct']:
        if c in env_df.columns:
            agg_df[f'{c}_mean_6h'] = grouped[c].transform(lambda x: x.rolling(6, min_periods=1).mean())
        
    return agg_df

env_feat = build_exogenous_features(lingkungan_df)
'''),

        new_markdown_cell('## 3. Injeksi Jangkar Memori, Topologi, & Sensor Anomaly Filter'),
        new_code_cell('''def prepare_local_dataset(base_df, env_feat, historical_tma_df, upstream_map):
    df = base_df.copy()
    
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['dayofyear'] = df['datetime'].dt.dayofyear
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24.0)
    
    df = pd.merge(df, env_feat, on=['datetime', 'nama_pos'], how='left')
    
    # 📌 V26 JANGKAR MEMORI
    df['datetime_minus_1_year'] = df['datetime'] - pd.DateOffset(years=1)
    historical_map = historical_tma_df.set_index(['nama_pos', 'datetime'])['tma_mdpl'].to_dict()
    df['tma_lag_365d'] = df.apply(lambda row: historical_map.get((row['nama_pos'], row['datetime_minus_1_year']), np.nan), axis=1)
    
    last_known_map = historical_tma_df.dropna(subset=['tma_mdpl']).sort_values('datetime').groupby('nama_pos').tail(1).set_index('nama_pos')['tma_mdpl'].to_dict()
    df['last_known_tma'] = df['nama_pos'].map(last_known_map)
    df = df.drop(columns=['datetime_minus_1_year'])
    
    # 📌 INJEKSI SPASIAL DARI HULU
    upstream_dfs = []
    for pos, upstreams in upstream_map.items():
        if len(upstreams) > 0:
            up_df = df[df['nama_pos'].isin(upstreams)]
            agg = up_df.groupby('datetime')[['api_48h', 'rain_sum_72h', 'last_known_tma']].mean().reset_index()
        else:
            up_df = df[df['nama_pos'] == pos]
            agg = up_df[['datetime', 'api_48h', 'rain_sum_72h', 'last_known_tma']].copy()
        
        agg = agg.rename(columns={
            'api_48h': 'upstream_api_48h',
            'rain_sum_72h': 'upstream_rain_sum_72h',
            'last_known_tma': 'upstream_last_known_tma'
        })
        agg['nama_pos'] = pos
        upstream_dfs.append(agg)
        
    all_upstream = pd.concat(upstream_dfs, ignore_index=True)
    df = pd.merge(df, all_upstream, on=['datetime', 'nama_pos'], how='left')
    
    return df

print("Menyusun Train Set (Frankenstein V30)...")
historical_tma = train_df[['datetime', 'nama_pos', 'tma_mdpl']].copy()
train_full = prepare_local_dataset(train_df, env_feat, historical_tma, upstream_map)

print("Menyusun Test Set (Frankenstein V30)...")
test_parsed = test_df.copy()
split_info = test_parsed['id'].str.split(' - ', n=1, expand=True)
test_parsed['datetime'] = pd.to_datetime(split_info[0])
test_parsed['nama_pos'] = split_info[1]
test_full = prepare_local_dataset(test_parsed, env_feat, historical_tma, upstream_map)

# Fillna Lags
for st in train_full['nama_pos'].unique():
    mask_tr = train_full['nama_pos'] == st
    train_full.loc[mask_tr, 'tma_lag_365d'] = train_full.loc[mask_tr, 'tma_lag_365d'].fillna(train_full.loc[mask_tr, 'last_known_tma'])
    
    mask_te = test_full['nama_pos'] == st
    test_full.loc[mask_te, 'tma_lag_365d'] = test_full.loc[mask_te, 'tma_lag_365d'].fillna(test_full.loc[mask_te, 'last_known_tma'])

train_full = train_full.dropna(subset=['tma_mdpl']).reset_index(drop=True)

# 📌 DETEKTOR ANOMALI SENSOR
print("Melakukan Filtering Sensor Anomaly (Bobot 0.05 untuk spikes acak)...")
def calculate_local_sample_weights(df, window=5, threshold=4.0):
    weights = []
    df_sorted = df.sort_values(['nama_pos', 'datetime'])
    rolling_med = df_sorted.groupby('nama_pos')['tma_mdpl'].transform(lambda x: x.rolling(window, center=True, min_periods=1).median())
    
    for tma, r_med in zip(df_sorted['tma_mdpl'], rolling_med):
        if abs(tma - r_med) > threshold:
            weights.append(0.05) 
        else:
            weights.append(1.0)
            
    df_sorted['sample_weight'] = weights
    return df_sorted.sort_index()

train_full = calculate_local_sample_weights(train_full)

drop_cols = ['datetime', 'nama_pos', 'tma_mdpl', 'id', 'sample_weight']
features = [c for c in train_full.columns if c not in drop_cols]
print(f"Total Fitur Raksasa per Pos: {len(features)}")
'''),

        new_markdown_cell('## 4. The Dual-Engine Training (Temporal CV)\\nMelatih XGBoost (Spike Catcher) & Random Forest (Stable Baseline) secara berbarengan untuk setiap stasiun!'),
        new_code_cell('''# Cutoff Waktu Validasi (Strict Temporal CV)
cutoffs = [
    ('2023-09-18', '2023-09-19', '2024-05-18'), # Fold 1
    ('2024-05-18', '2024-05-19', '2025-01-18'), # Fold 2
    ('2025-01-18', '2025-01-19', '2025-09-18')  # Fold 3
]

xgb_final_models = {}
rf_final_models = {}

dt_full = train_full['datetime']
all_stations = train_full['nama_pos'].unique()

print("Mulai Pelatihan Dual-Engine (3 Temporal Folds per Pos)\\n" + "="*60)

for stasiun in all_stations:
    df_pos = train_full[train_full['nama_pos'] == stasiun].reset_index(drop=True)
    X = df_pos[features]
    
    # 📌 KUNCI BUGFIX: Tweedie tidak bisa menerima nilai negatif (Kasus Ketonggo)
    y = df_pos['tma_mdpl'].clip(lower=0) 
    w = df_pos['sample_weight']
    dt = df_pos['datetime']
    
    hybrid_fold_rmse = []
    
    print(f"🌊 STASIUN: {stasiun}")
    
    for i, (tr_end, val_start, val_end) in enumerate(cutoffs):
        train_mask = dt <= pd.to_datetime(tr_end)
        val_mask = (dt >= pd.to_datetime(val_start)) & (dt <= pd.to_datetime(val_end))
        
        # Jika data fold tidak cukup (beberapa stasiun baru aktif di 2024), lewati fold ini
        if train_mask.sum() < 100 or val_mask.sum() < 10:
            continue
            
        X_tr, y_tr, w_tr = X[train_mask], y[train_mask], w[train_mask]
        X_va, y_va = X[val_mask], y[val_mask]
        
        # --- ENGINE 1: XGBoost (Tweedie) ---
        xgb_model = xgb.XGBRegressor(
            objective='reg:tweedie',
            tweedie_variance_power=1.5,
            eval_metric=['tweedie-nloglik@1.5', 'rmse'], # Nampilin dua metrics sekaligus
            tree_method='hist',
            device='cuda',
            learning_rate=0.03,
            max_depth=6,
            n_estimators=1000,
            early_stopping_rounds=30,
            random_state=42
        )
        xgb_model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_va, y_va)], verbose=False)
        
        # Ambil best iteration info
        best_iter = xgb_model.best_iteration
        best_tweedie = xgb_model.evals_result()['validation_0']['tweedie-nloglik@1.5'][best_iter]
        best_rmse_xgb = xgb_model.evals_result()['validation_0']['rmse'][best_iter]
        
        # --- ENGINE 2: Random Forest GPU (MSE) ---
        rf_model = xgb.XGBRFRegressor(
            n_estimators=200,
            max_depth=10,
            tree_method='hist',
            device='cuda',
            learning_rate=1.0,
            random_state=42
        )
        rf_model.fit(X_tr, y_tr)
        
        # --- EVALUASI HYBRID (BLENDER 50:50) ---
        pred_xgb = xgb_model.predict(X_va)
        pred_rf = rf_model.predict(X_va)
        pred_hybrid = (0.5 * pred_xgb) + (0.5 * pred_rf)
        
        fold_rmse = root_mean_squared_error(y_va, pred_hybrid)
        hybrid_fold_rmse.append(fold_rmse)
        
        print(f"   Fold {i+1} | XGB BestIter: {best_iter} (Tweedie: {best_tweedie:.4f}, RMSE: {best_rmse_xgb:.4f}) | Hybrid Final RMSE: {fold_rmse:.4f}")
    
    if len(hybrid_fold_rmse) > 0:
        print(f"   >>> Rata-rata Hybrid RMSE: {np.mean(hybrid_fold_rmse):.4f}\\n")
    
    # 📌 MELATIH MODEL FINAL UNTUK INFERENSI TEST SET (100% DATA LOKAL)
    final_xgb = xgb.XGBRegressor(
        objective='reg:tweedie', tweedie_variance_power=1.5, tree_method='hist', device='cuda',
        learning_rate=0.03, max_depth=6, n_estimators=400, random_state=42
    )
    final_xgb.fit(X, y, sample_weight=w, verbose=False)
    xgb_final_models[stasiun] = final_xgb
    
    final_rf = xgb.XGBRFRegressor(n_estimators=300, max_depth=12, tree_method='hist', device='cuda', learning_rate=1.0, random_state=42)
    final_rf.fit(X, y)
    rf_final_models[stasiun] = final_rf

print("✅ Seluruh 60 Model Dual-Engine Selesai Dilatih!")
'''),

        new_markdown_cell('## 5. Inferensi Test Set (Blending 50:50)'),
        new_code_cell('''print("Memprediksi Test Set (Hybrid Blending)...")
test_full['tma_mdpl'] = np.nan
test_full = test_full.fillna(method='ffill').fillna(0) # Safety fill untuk fitur NaN

for stasiun in test_full['nama_pos'].unique():
    mask = test_full['nama_pos'] == stasiun
    if mask.sum() == 0: continue
    
    X_test_pos = test_full.loc[mask, features]
    
    # Dual Predict
    pred_xgb = xgb_final_models[stasiun].predict(X_test_pos)
    pred_rf = rf_final_models[stasiun].predict(X_test_pos)
    
    # 50:50 Blender
    test_full.loc[mask, 'tma_mdpl'] = (0.5 * pred_xgb) + (0.5 * pred_rf)

sub_pred = test_full[['id', 'tma_mdpl']].copy()
sub = pd.merge(sample_sub[['id']], sub_pred, on='id', how='left')
sub['tma_mdpl'] = sub['tma_mdpl'].fillna(method='ffill').fillna(method='bfill')

display(sub.head())
sub.to_csv('submission_v30_ultimate_frankenstein.csv', index=False)
print("\\n🚀 File 'submission_v30_ultimate_frankenstein.csv' BERHASIL DIBUAT!")
''')
    ]
    
    nb.cells.extend(cells)
    with open('V30_The_Ultimate_Frankenstein.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == '__main__':
    create_v30_notebook()
    print("Notebook V30_The_Ultimate_Frankenstein.ipynb berhasil dibuat!")
