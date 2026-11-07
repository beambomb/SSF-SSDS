import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

def create_v28_notebook():
    nb = new_notebook()
    
    cells = [
        new_markdown_cell('# 🧬 V28 THE ULTIMATE PHYSICS-INFUSED LOCAL MODELS 🧬\\nSelamat datang di arsitektur pamungkas! Kita kembali ke 30 Model XGBoost Lokal (Tweedie Loss) yang terbukti sakti di Leaderboard.\\nNamun kali ini, mereka dipersenjatai dengan fitur *God-Tier*:\\n1. **Fisika Air (EWM & Dew Point)** dari arsitektur Global.\\n2. **Sensor Anomaly Downweighting** (Bobot 0.05 untuk lonjakan aneh) agar XGBoost lokal kebal dari sensor rusak.\\n3. **3 Jangkar Memori** (`last_known_tma`, `tma_lag_365d`, `upstream_tma_lag`) dikembalikan agar *baseline* air tidak melayang.\\n4. **Random K-Fold** diaktifkan kembali karena XGBoost Lokal butuh sedikit kebocoran cuaca untuk stabil di Kaggle.'),
        
        new_markdown_cell('## 1. Persiapan Environment & Data'),
        new_code_cell('''!pip install -q xgboost scikit-learn optuna folium
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import optuna
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold

plt.style.use('seaborn-v0_8-whitegrid')

# Auto-Detect Path
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

        new_markdown_cell('## 2. Pemetaan Topologi Spasial & Ekstraksi Fisika Air (EWM)'),
        new_code_cell('''print("Membangun Peta Topologi Sungai...")

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

print("Mengekstraksi Fitur Fisika Air (API & Dew-Point)...")
lingkungan_df = lingkungan_df.sort_values(['nama_pos', 'datetime']).reset_index(drop=True)

def build_exogenous_features(env_df):
    agg_df = env_df[['datetime', 'nama_pos']].copy()
    grouped = env_df.groupby('nama_pos')
    
    # 📌 1. DEW-POINT DEPRESSION (Rumus Evaporasi Magnus)
    if 'temperature_c' in env_df.columns and 'humidity_pct' in env_df.columns:
        env_df['dew_point'] = env_df['temperature_c'] - ((100 - env_df['humidity_pct']) / 5.0)
        env_df['dew_point_dep'] = env_df['temperature_c'] - env_df['dew_point']
        agg_df['dew_point_dep_mean_6h'] = grouped['dew_point_dep'].transform(lambda x: x.rolling(6, min_periods=1).mean())
        
    # 📌 2. ANTECEDENT PRECIPITATION INDEX (EWM)
    if 'rainfall_mm' in env_df.columns:
        agg_df['api_12h'] = grouped['rainfall_mm'].transform(lambda x: x.ewm(halflife=12, min_periods=1).mean())
        agg_df['api_48h'] = grouped['rainfall_mm'].transform(lambda x: x.ewm(halflife=48, min_periods=1).mean())
        agg_df['api_168h'] = grouped['rainfall_mm'].transform(lambda x: x.ewm(halflife=168, min_periods=1).mean())
        agg_df['rain_max_24h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(24, min_periods=1).max())
        
    if 'soil_moisture_0_7cm' in env_df.columns:
        agg_df['soil_mean_6h'] = grouped['soil_moisture_0_7cm'].transform(lambda x: x.rolling(6, min_periods=1).mean())
        agg_df['soil_mean_72h'] = grouped['soil_moisture_0_7cm'].transform(lambda x: x.rolling(72, min_periods=1).mean())
        
    for c in ['nino_34', 'mjo_phase']:
        if c in env_df.columns:
            agg_df[f'{c}_mean_720h'] = grouped[c].transform(lambda x: x.rolling(720, min_periods=1).mean())
            
    for c in ['temperature_c', 'wind_speed_kmh', 'surface_pressure_hpa', 'humidity_pct']:
        if c in env_df.columns:
            agg_df[f'{c}_mean_6h'] = grouped[c].transform(lambda x: x.rolling(6, min_periods=1).mean())
        
    # 📌 3. FITUR GLOBAL BASIN
    basin_env = env_df.groupby('datetime').agg({'rainfall_mm': 'mean', 'soil_moisture_0_7cm': 'mean'}).reset_index()
    basin_env['basin_api_72h'] = basin_env['rainfall_mm'].ewm(halflife=72, min_periods=1).mean()
    basin_env['basin_api_168h'] = basin_env['rainfall_mm'].ewm(halflife=168, min_periods=1).mean()
    basin_env['basin_soil_mean_168h'] = basin_env['soil_moisture_0_7cm'].rolling(168, min_periods=1).mean()
    
    basin_features = basin_env[['datetime', 'basin_api_72h', 'basin_api_168h', 'basin_soil_mean_168h']]
    agg_df = pd.merge(agg_df, basin_features, on='datetime', how='left')

    return agg_df

env_feat = build_exogenous_features(lingkungan_df)
'''),

        new_markdown_cell('## 3. Restorasi Fitur Jangkar Memori & Pembobotan Anomali Sensor (0.05)'),
        new_code_cell('''def prepare_local_dataset(base_df, env_feat, historical_tma_df, upstream_map):
    df = base_df.copy()
    
    # Waktu
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['dayofyear'] = df['datetime'].dt.dayofyear
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12.0)
    
    df = pd.merge(df, env_feat, on=['datetime', 'nama_pos'], how='left')
    
    # 📌 REKOVERI JANGKAR (V26)
    df['datetime_minus_1_year'] = df['datetime'] - pd.DateOffset(years=1)
    historical_map = historical_tma_df.set_index(['nama_pos', 'datetime'])['tma_mdpl'].to_dict()
    df['tma_lag_365d'] = df.apply(lambda row: historical_map.get((row['nama_pos'], row['datetime_minus_1_year']), np.nan), axis=1)
    
    last_known_map = historical_tma_df.dropna(subset=['tma_mdpl']).sort_values('datetime').groupby('nama_pos').tail(1).set_index('nama_pos')['tma_mdpl'].to_dict()
    df['last_known_tma'] = df['nama_pos'].map(last_known_map)
    df = df.drop(columns=['datetime_minus_1_year'])
    
    # 📌 SUNTIKAN 4 FITUR HULU (EWM + JANGKAR)
    print("  -> Menyuntikkan 4 Fitur Fisika Aliran Hulu...")
    upstream_dfs = []
    for pos, upstreams in upstream_map.items():
        if len(upstreams) > 0:
            up_df = df[df['nama_pos'].isin(upstreams)]
            agg = up_df.groupby('datetime')[['api_48h', 'dew_point_dep_mean_6h', 'last_known_tma', 'tma_lag_365d']].mean().reset_index()
        else:
            up_df = df[df['nama_pos'] == pos]
            agg = up_df[['datetime', 'api_48h', 'dew_point_dep_mean_6h', 'last_known_tma', 'tma_lag_365d']].copy()
        
        agg = agg.rename(columns={
            'api_48h': 'upstream_api_48h',
            'dew_point_dep_mean_6h': 'upstream_dew_point_dep',
            'last_known_tma': 'upstream_last_known_tma',
            'tma_lag_365d': 'upstream_tma_lag_365d'
        })
        agg['nama_pos'] = pos
        upstream_dfs.append(agg)
        
    all_upstream = pd.concat(upstream_dfs, ignore_index=True)
    df = pd.merge(df, all_upstream, on=['datetime', 'nama_pos'], how='left')
    
    return df

print("Menyusun Train Set (V28)...")
historical_tma = train_df[['datetime', 'nama_pos', 'tma_mdpl']].copy()
train_full = prepare_local_dataset(train_df, env_feat, historical_tma, upstream_map)

print("Menyusun Test Set (V28)...")
test_parsed = test_df.copy()
split_info = test_parsed['id'].str.split(' - ', n=1, expand=True)
test_parsed['datetime'] = pd.to_datetime(split_info[0])
test_parsed['nama_pos'] = split_info[1]
test_full = prepare_local_dataset(test_parsed, env_feat, historical_tma, upstream_map)

for st in train_full['nama_pos'].unique():
    mask_tr = train_full['nama_pos'] == st
    train_full.loc[mask_tr, 'tma_lag_365d'] = train_full.loc[mask_tr, 'tma_lag_365d'].fillna(train_full.loc[mask_tr, 'last_known_tma'])
    train_full.loc[mask_tr, 'upstream_tma_lag_365d'] = train_full.loc[mask_tr, 'upstream_tma_lag_365d'].fillna(train_full.loc[mask_tr, 'upstream_last_known_tma'])
    
    mask_te = test_full['nama_pos'] == st
    test_full.loc[mask_te, 'tma_lag_365d'] = test_full.loc[mask_te, 'tma_lag_365d'].fillna(test_full.loc[mask_te, 'last_known_tma'])
    test_full.loc[mask_te, 'upstream_tma_lag_365d'] = test_full.loc[mask_te, 'upstream_tma_lag_365d'].fillna(test_full.loc[mask_te, 'upstream_last_known_tma'])

train_full = train_full.dropna(subset=['tma_mdpl']).reset_index(drop=True)

# 📌 DETEKTOR SENSOR RUSAK (LOCAL DOWNWEIGHTING)
print("Mendeteksi Spike Sensor Palsu (Anomaly Downweighting)...")
def calculate_local_sample_weights(df, window=5, threshold=4.0):
    weights = []
    df_sorted = df.sort_values(['nama_pos', 'datetime'])
    rolling_med = df_sorted.groupby('nama_pos')['tma_mdpl'].transform(lambda x: x.rolling(window, center=True, min_periods=1).median())
    
    anom_count = 0
    for tma, r_med in zip(df_sorted['tma_mdpl'], rolling_med):
        # Jika tiba-tiba melompat drastis (4 meter) dari rata-rata tetangganya
        if abs(tma - r_med) > threshold:
            weights.append(0.05) 
            anom_count += 1
        else:
            weights.append(1.0)
            
    df_sorted['sample_weight'] = weights
    print(f"-> Terdeteksi {anom_count} baris data spike (Bobot disunat menjadi 0.05)")
    return df_sorted.sort_index()

train_full = calculate_local_sample_weights(train_full)

drop_cols = ['datetime', 'nama_pos', 'tma_mdpl', 'id', 'sample_weight']
features = [c for c in train_full.columns if c not in drop_cols]
print(f"Total Fitur per Pos: {len(features)}")
'''),

        new_markdown_cell('## 4. Pelatihan 30 Model Lokal + Optuna Tweedie (K-Fold)\\nSetiap stasiun dicari parameter terbaiknya sendiri menggunakan GPU!'),
        new_code_cell('''optuna.logging.set_verbosity(optuna.logging.ERROR)
N_TRIALS = 15
best_params_per_pos = {}
pos_models = {}

print(f"Memulai Optuna untuk 30 Stasiun @ {N_TRIALS} Trials/pos (Menggunakan K-Fold)...\\n")

for stasiun in train_full['nama_pos'].unique():
    print(f"Memproses stasiun: {stasiun}...")
    
    df_pos = train_full[train_full['nama_pos'] == stasiun].reset_index(drop=True)
    X = df_pos[features]
    # KUNCI BUGFIX: Tweedie tidak bisa menerima nilai negatif.
    # Stasiun 'Ketonggo' memiliki TMA -0.059 meter. Kita clip ke 0.
    y = df_pos['tma_mdpl'].clip(lower=0)
    w = df_pos['sample_weight']
    
    def objective(trial):
        params = {
            'objective': 'reg:tweedie',
            'tweedie_variance_power': trial.suggest_float('tweedie_variance_power', 1.1, 1.9),
            'eval_metric': 'mae',
            'tree_method': 'hist',
            'device': 'cuda',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'n_estimators': 800,
            'random_state': 42
        }
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        for tr_idx, va_idx in kf.split(X):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
            w_tr = w.iloc[tr_idx]
            
            model = xgb.XGBRegressor(**params, early_stopping_rounds=20)
            model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_va, y_va)], verbose=False)
            
            preds = model.predict(X_va)
            cv_scores.append(root_mean_squared_error(y_va, preds))
            
        return np.mean(cv_scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS)
    
    best_p = study.best_params
    best_p['objective'] = 'reg:tweedie'
    best_p['eval_metric'] = 'mae'
    best_p['tree_method'] = 'hist'
    best_p['device'] = 'cuda'
    best_p['n_estimators'] = 800
    best_p['random_state'] = 42
    
    best_params_per_pos[stasiun] = best_p
    
    # Train Final Model Lokal
    final_local_model = xgb.XGBRegressor(**best_p)
    final_local_model.fit(X, y, sample_weight=w, verbose=False)
    pos_models[stasiun] = final_local_model
    
    print(f"  -> Best Local RMSE: {study.best_value:.4f}")

print("\\n✅ Seluruh 30 Model V28 Selesai Dilatih!")
'''),

        new_markdown_cell('## 5. Eksekusi Prediksi Test Set V28'),
        new_code_cell('''print("Memprediksi Test Set per Stasiun...")
test_full['tma_mdpl'] = np.nan

for stasiun in test_full['nama_pos'].unique():
    mask = test_full['nama_pos'] == stasiun
    X_test_pos = test_full.loc[mask, features]
    
    model = pos_models[stasiun]
    preds = model.predict(X_test_pos)
    
    test_full.loc[mask, 'tma_mdpl'] = preds

sub_pred = test_full[['id', 'tma_mdpl']].copy()
sub = pd.merge(sample_sub[['id']], sub_pred, on='id', how='left')
sub['tma_mdpl'] = sub['tma_mdpl'].fillna(method='ffill').fillna(method='bfill')

display(sub.head())
sub.to_csv('submission_v28_ultimate_hybrid.csv', index=False)
print("\\n🚀 File 'submission_v28_ultimate_hybrid.csv' BERHASIL DIBUAT!")
'''),
        new_markdown_cell('## 6. Visualisasi Inferensi 30 Model Lokal'),
        new_code_cell('''import ipywidgets as widgets
from IPython.display import display

print("Memproses prediksi pada data latih untuk visualisasi...")

# Memprediksi Train Set menggunakan 30 model terpisah
tma_train_preds = np.zeros(len(train_full))
for stasiun in train_full['nama_pos'].unique():
    mask = train_full['nama_pos'] == stasiun
    X_train_pos = train_full.loc[mask, features]
    tma_train_preds[mask] = pos_models[stasiun].predict(X_train_pos)

train_full_vis = train_full.copy()
train_full_vis['tma_pred'] = tma_train_preds

def plot_actual_vs_predict_v28(stasiun):
    df_plot = train_full_vis[train_full_vis['nama_pos'] == stasiun].sort_values('datetime')
    df_plot = df_plot[df_plot['datetime'] >= '2024-09-01'] 
    
    plt.figure(figsize=(16, 6))
    
    # Plot Garis Actual (Biru)
    plt.plot(df_plot['datetime'], df_plot['tma_mdpl'], label='Actual TMA (Asli)', 
             color='royalblue', linewidth=2.5, alpha=0.8)
             
    # Plot Garis Predicted (Oranye Putus-putus)
    plt.plot(df_plot['datetime'], df_plot['tma_pred'], label='Predicted TMA (Model Lokal)', 
             color='darkorange', linewidth=2, linestyle='--')
    
    plt.title(f'Kinerja V28 Ultimate Local - Stasiun: {stasiun}', fontsize=16, fontweight='bold')
    plt.xlabel('Waktu', fontsize=12)
    plt.ylabel('Tinggi Muka Air (Meter)', fontsize=12)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()

daftar_pos = sorted(train_full_vis['nama_pos'].unique())
dropdown = widgets.Dropdown(options=daftar_pos, description='Pilih Pos:', style={'description_width': 'initial'})
widgets.interact(plot_actual_vs_predict_v28, stasiun=dropdown)
''')
    ]
    
    nb.cells.extend(cells)
    with open('V28_Ultimate_Hybrid.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == '__main__':
    create_v28_notebook()
    print("Notebook V28_Ultimate_Hybrid.ipynb berhasil dibuat!")
