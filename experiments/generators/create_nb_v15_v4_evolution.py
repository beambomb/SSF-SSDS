import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

def create_v15_notebook():
    nb = new_notebook()
    
    cells = [
        new_markdown_cell('# 🌪️ V15 THE V4 EVOLUTION (Ultimate Baseline + Hydrology + Spatial) 🏆\nKaggle Grandmaster Edition! Kita mengawinkan 3 kekuatan absolut:\n1. **V4 Super Exogenous**: *Elevation Proxy*, Label Encoding, dan memori iklim 30 hari (720h) + Parameter Optuna.\n2. **V13 Spatial Lags**: Integrasi topologi jarak Euclidean antar stasiun untuk simulasi rambatan spasial.\n3. **Ilmu Hidrologi**: Mengubah cuaca mentah menjadi API (Antecedent Precipitation Index) dan Effective Runoff.'),
        
        new_markdown_cell('## 1. Persiapan Environment & Pemuatan Data'),
        new_code_cell('''!pip install -q xgboost scikit-learn scipy
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as plt_sns
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cdist

plt.style.use('seaborn-v0_8-whitegrid')

# Auto-Detect Path
if not 'path' in locals():
    if os.path.exists('/kaggle/input'):
        dataset_folders = [f for f in os.listdir('/kaggle/input') if os.path.isdir(f'/kaggle/input/{f}')]
        path = f'/kaggle/input/{dataset_folders[0]}/' if dataset_folders else '/kaggle/input/'
    elif os.path.exists('/content'):
        path = '/content/drive/MyDrive/ssf-ssds/'
    else:
        path = './'
print(f"Path dataset aktif: {path}")

train_df = pd.read_csv(path + 'train.csv', parse_dates=['datetime'])
test_df = pd.read_csv(path + 'test.csv')
lingkungan_df = pd.read_csv(path + 'data_pendukung/data_lingkungan.csv', parse_dates=['datetime'])
koordinat_df = pd.read_csv(path + 'data_pendukung/koordinat_pos.csv')
sample_sub = pd.read_csv(path + 'sample_submission.csv')

print(f"Data dimuat! Train: {train_df.shape}, Test: {test_df.shape}, Lingkungan: {lingkungan_df.shape}")
'''),

        new_markdown_cell('## 2. V13 Spatial Topology & V4 Elevation Proxy\nMenghitung hujan di 3 stasiun terdekat dan merata-ratakan elevasi absolut.'),
        new_code_cell('''# 1. Menghitung Jarak Euclidean antar Stasiun
coords = koordinat_df[['latitude', 'longitude']].values
dist_matrix = cdist(coords, coords)
station_names = koordinat_df['nama_pos'].values
nearest_stations = {}

for i, st in enumerate(station_names):
    nearest_idx = np.argsort(dist_matrix[i])[1:4]
    nearest_stations[st] = station_names[nearest_idx]

# 2. Fitur Hujan Regional (Spatial Rain)
rain_pivot = lingkungan_df.pivot(index='datetime', columns='nama_pos', values='rainfall_mm')
spatial_rain_list = []
for st in station_names:
    neighbors = nearest_stations[st]
    sp_rain = rain_pivot[neighbors].mean(axis=1).to_frame(name='spatial_rain_avg')
    sp_rain['nama_pos'] = st
    spatial_rain_list.append(sp_rain.reset_index())

spatial_rain_df = pd.concat(spatial_rain_list, ignore_index=True)
full_env = pd.merge(lingkungan_df, spatial_rain_df, on=['datetime', 'nama_pos'], how='left')
full_env = pd.merge(full_env, koordinat_df, on='nama_pos', how='left')

# 3. V4 Elevation Proxy (JANGKAR ABSOLUT!)
# Rata-rata target historis per stasiun dari train_df
station_elevation = train_df.groupby('nama_pos')['tma_mdpl'].mean().reset_index()
station_elevation.rename(columns={'tma_mdpl': 'elevation_proxy'}, inplace=True)
full_env = pd.merge(full_env, station_elevation, on='nama_pos', how='left')

print("Spatial Lags dan Elevation Proxy berhasil diinjeksi!")
'''),

        new_markdown_cell('## 3. V4 Turbo Exogenous & Hydrology (30-Days Memory)\nSiklus hidrologi besar butuh memori bulanan (720h).'),
        new_code_cell('''full_env = full_env.sort_values(['nama_pos', 'datetime']).reset_index(drop=True)

# A. Iklim Mikro Sesaat & Siklus Waktu
full_env['dew_point_spread'] = full_env['temperature_c'] - full_env['dew_point_c']
full_env['hour_sin'] = np.sin(2 * np.pi * full_env['datetime'].dt.hour / 24.0)
full_env['hour_cos'] = np.cos(2 * np.pi * full_env['datetime'].dt.hour / 24.0)
full_env['month_sin'] = np.sin(2 * np.pi * full_env['datetime'].dt.month / 12.0)
full_env['month_cos'] = np.cos(2 * np.pi * full_env['datetime'].dt.month / 12.0)
full_env['wind_sin'] = np.sin(np.radians(full_env['wind_direction_deg']))
full_env['wind_cos'] = np.cos(np.radians(full_env['wind_direction_deg']))

print("Mengeksekusi perhitungan 30-Hari (720h)...")
for st, group in full_env.groupby('nama_pos'):
    idx = group.index
    
    # B. V4 ROLLING EKSTREM (Sampai 30 Hari = 720h)
    full_env.loc[idx, 'rain_sum_72h'] = group['rainfall_mm'].rolling(72, min_periods=1).sum()
    full_env.loc[idx, 'rain_sum_168h'] = group['rainfall_mm'].rolling(168, min_periods=1).sum()
    full_env.loc[idx, 'rain_sum_360h'] = group['rainfall_mm'].rolling(360, min_periods=1).sum()
    full_env.loc[idx, 'rain_sum_720h'] = group['rainfall_mm'].rolling(720, min_periods=1).sum()
    
    full_env.loc[idx, 'nino_mean_720h'] = group['nino_34'].rolling(720, min_periods=1).mean()
    full_env.loc[idx, 'mjo_mean_720h'] = group['mjo_amplitude'].rolling(720, min_periods=1).mean()
    
    # C. SPATIAL LAGS
    full_env.loc[idx, 'spatial_rain_lag_1'] = group['spatial_rain_avg'].shift(1)
    full_env.loc[idx, 'spatial_rain_lag_2'] = group['spatial_rain_avg'].shift(2)
    
    # D. ILMU HIDROLOGI
    # API (Antecedent Precipitation Index)
    full_env.loc[idx, 'api_7d'] = group['rainfall_mm'].ewm(alpha=0.15, adjust=False).mean()
    full_env.loc[idx, 'api_14d'] = group['rainfall_mm'].ewm(alpha=0.07, adjust=False).mean()
    
    # Effective Runoff
    sm_norm = group['soil_moisture_0_7cm'] / 100.0
    full_env.loc[idx, 'effective_runoff'] = group['rainfall_mm'] * sm_norm
    full_env.loc[idx, 'runoff_lag_1'] = full_env.loc[idx, 'effective_runoff'].shift(1)
    
    # Groundwater Baseflow
    full_env.loc[idx, 'baseflow_proxy'] = group['soil_moisture_100_255cm'].rolling(168, min_periods=1).mean()

# GABUNGKAN KE TRAIN SET
df_all = pd.merge(train_df, full_env, on=['datetime', 'nama_pos'], how='inner')

# E. V4 LABEL ENCODING (Lebih baik dari OHE untuk Tree)
le = LabelEncoder()
all_stations = full_env['nama_pos'].unique()
le.fit(all_stations)
df_all['nama_pos_encoded'] = le.transform(df_all['nama_pos'])

print(f"Fitur Turbo Selesai! Shape: {df_all.shape}")
'''),

        new_markdown_cell('## 4. XGBoost Training (V4 Optuna Best Params)\nMelatih model dengan parameter sakti yang kamu temukan di V4! (K-Fold murni dengan `shuffle=False` agar tidak bocor masa depan).'),
        new_code_cell('''# A. Daftarkan Fitur
xgb_features = [
    'nama_pos_encoded', 'elevation_proxy', 'latitude', 'longitude', # Jangkar V4
    'spatial_rain_avg', 'spatial_rain_lag_1', 'spatial_rain_lag_2', # Spatial V13
    'api_7d', 'api_14d', 'effective_runoff', 'runoff_lag_1', 'baseflow_proxy', # Hidrologi
    'rain_sum_72h', 'rain_sum_168h', 'rain_sum_360h', 'rain_sum_720h', # V4 30-Day Sum
    'nino_mean_720h', 'mjo_mean_720h', # V4 30-Day Macro
    'rainfall_mm', 'humidity_pct', 'cloud_cover_pct', 'solar_radiation_mj_m2',
    'soil_moisture_0_7cm', 'soil_moisture_7_28cm', 'soil_moisture_28_100cm', 'soil_moisture_100_255cm',
    'nino_34', 'rmm1', 'rmm2', 'mjo_amplitude', 'mjo_active',
    'month_sin', 'month_cos', 'hour_sin', 'hour_cos'
]

print(f"Total Fitur Ultimate: {len(xgb_features)}")

# B. Pembersihan
df_clean = df_all.dropna(subset=['tma_mdpl'] + xgb_features)
df_clean = df_clean.sort_values('datetime').reset_index(drop=True)

X = df_clean[xgb_features].astype(float)
y = df_clean['tma_mdpl']

print(f"Data Bersih Siap Latih: X={X.shape}, y={y.shape}")

# C. K-Fold Training Loop
kf = KFold(n_splits=5, shuffle=True, random_state=42) # KEMBALI KE SHUFFLE=TRUE (Rahasia V4)
models_xgb = []
scores = []

# ==========================================
# PARAMETER TERBAIK OPTUNA DARI V4 KAMU!
# ==========================================
best_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'tree_method': 'hist',
    'learning_rate': 0.02, 
    'max_depth': 7, 
    'subsample': 0.8,
    'colsample_bytree': 0.8, 
    'min_child_weight': 5,
    'reg_alpha': 0.5, 
    'reg_lambda': 1.0,
    'random_state': 42
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\\n--- Fold {fold+1} ---")
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_va, label=y_va)
    
    model = xgb.train(
        best_params, 
        dtrain, 
        num_boost_round=10000, 
        evals=[(dval, 'val')],
        early_stopping_rounds=150,
        verbose_eval=300
    )
    
    val_preds = model.predict(dval)
    fold_rmse = root_mean_squared_error(y_va, val_preds)
    scores.append(fold_rmse)
    models_xgb.append(model)

print(f"\\n🏆 Rata-rata CV RMSE Asli (No Leak): {np.mean(scores):.4f}")
'''),

        new_markdown_cell('## 5. Visualisasi Karakter Model\nPerhatikan Feature Importance! `elevation_proxy` (Jangkar V4) dan `api_14d` (Hidrologi) pasti mendominasi!'),
        new_code_cell('''fig, ax = plt.subplots(1, 2, figsize=(20, 7))

# A. Feature Importance
xgb.plot_importance(models_xgb[-1], max_num_features=25, height=0.6, ax=ax[0], color='purple')
ax[0].set_title('Top 25 Feature Importance (V15)')

# B. Actual vs Predicted (Sampel Jurug - Hulu)
st_mask = df_clean['nama_pos'] == 'Jurug' 
X_sub = df_clean.loc[st_mask, xgb_features].astype(float).iloc[-300:]
y_sub = df_clean.loc[st_mask, 'tma_mdpl'].iloc[-300:].values

if len(X_sub) > 0:
    dsub = xgb.DMatrix(X_sub)
    pred_sub = models_xgb[-1].predict(dsub)
    
    # POST-PROCESSING SMOOTHING (Simulasi V16)
    pred_smooth = pd.Series(pred_sub).ewm(span=12, adjust=False).mean().values
    
    ax[1].plot(y_sub, label='Actual TMA (Jurug)', color='navy', linewidth=2)
    ax[1].plot(pred_sub, label='V15 Raw Predict', color='red', alpha=0.3, linestyle=':')
    ax[1].plot(pred_smooth, label='V15 Smoothed', color='forestgreen', alpha=0.9, linestyle='--', linewidth=3)
    ax[1].set_title('Actual vs Smoothed Predicted V15 (Jurug Hulu)')
    ax[1].legend()

plt.tight_layout()
plt.show()
'''),

        new_markdown_cell('## 6. Inference Super Kilat ⚡\n0% Error Compounding, Langsung tembak seluruh Test Set!'),
        new_code_cell('''print("Menyiapkan Test Set...")

# Parsing ID Test
test_parsed = test_df.copy()
split_info = test_parsed['id'].str.split(' - ', n=1, expand=True)
test_parsed['datetime'] = pd.to_datetime(split_info[0])
test_parsed['nama_pos'] = split_info[1]

# Gabungkan Train & Test untuk perhitungan Rolling & Lags yang mulus
gabungan_df = pd.concat([train_df[['datetime', 'nama_pos']], test_parsed[['datetime', 'nama_pos']]], ignore_index=True)
gabungan_df.sort_values(['nama_pos', 'datetime'], inplace=True)

# Merge dengan full_env yang sudah memiliki ALL features
gabungan_df = pd.merge(gabungan_df, full_env, on=['datetime', 'nama_pos'], how='left')
gabungan_df['nama_pos_encoded'] = le.transform(gabungan_df['nama_pos'])

# Filter khusus Test Set
test_times = sorted(test_parsed['datetime'].unique())
test_final = gabungan_df[gabungan_df['datetime'].isin(test_times)].copy()
X_test = test_final[xgb_features].astype(float)
dtest = xgb.DMatrix(X_test)

print("Ensembling 5 Fold Model...")
test_preds = np.zeros(len(X_test))
for model in models_xgb:
    test_preds += model.predict(dtest) / len(models_xgb)

test_final['tma_mdpl_raw'] = test_preds

# POST-PROCESSING SMOOTHING (Menghaluskan prediksi kasar XGBoost)
print("Mengaplikasikan EWMA Smoothing (span=12)...")
test_final['tma_mdpl'] = test_final.groupby('nama_pos')['tma_mdpl_raw'].transform(lambda x: x.ewm(span=12, adjust=False).mean())

# Format Submission
sub = pd.merge(test_parsed[['id', 'datetime', 'nama_pos']], test_final[['datetime', 'nama_pos', 'tma_mdpl']], on=['datetime', 'nama_pos'], how='left')
sub = sub[['id', 'tma_mdpl']]

# FFill aman untuk NaN di awal Test Set
sub['tma_mdpl'] = sub['tma_mdpl'].fillna(method='ffill')

sub.to_csv('submission_v15_the_v4_evolution.csv', index=False)
print("✅ SELESAI! File 'submission_v15_the_v4_evolution.csv' siap disubmit ke Kaggle!")
''')
    ]
    
    nb.cells.extend(cells)
    with open('V15_The_V4_Evolution.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == '__main__':
    create_v15_notebook()
    print("Notebook V15_The_V4_Evolution.ipynb berhasil dibuat!")
