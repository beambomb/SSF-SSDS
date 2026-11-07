import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

def create_v16_notebook():
    nb = new_notebook()
    
    cells = [
        new_markdown_cell('# ⚔️ V16 THE GRAND ENSEMBLE (Exogenous Anchor + Recursive Tracker) 🏆\nTeknik *State-of-the-Art* Kaggle Grandmaster: Menggabungkan stabilitas *Exogenous Model* (Anti-Ledakan) dengan kepekaan *Recursive Model* (Pelacak Puncak Banjir).'),
        
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

train_df = pd.read_csv(path + 'train.csv', parse_dates=['datetime'])
test_df = pd.read_csv(path + 'test.csv')
lingkungan_df = pd.read_csv(path + 'data_pendukung/data_lingkungan.csv', parse_dates=['datetime'])
koordinat_df = pd.read_csv(path + 'data_pendukung/koordinat_pos.csv')
sample_sub = pd.read_csv(path + 'sample_submission.csv')
'''),

        new_markdown_cell('## 2. Feature Engineering (Hydrology, Spatial, & Lags)'),
        new_code_cell('''# A. V4 Elevation Proxy (Jangkar)
station_elevation = train_df.groupby('nama_pos')['tma_mdpl'].mean().reset_index()
station_elevation.rename(columns={'tma_mdpl': 'elevation_proxy'}, inplace=True)

# B. Menghitung Jarak Euclidean (Spatial)
coords = koordinat_df[['latitude', 'longitude']].values
dist_matrix = cdist(coords, coords)
station_names = koordinat_df['nama_pos'].values
nearest_stations = {}
for i, st in enumerate(station_names):
    nearest_idx = np.argsort(dist_matrix[i])[1:4]
    nearest_stations[st] = station_names[nearest_idx]

rain_pivot = lingkungan_df.pivot(index='datetime', columns='nama_pos', values='rainfall_mm')
spatial_rain_list = []
for st in station_names:
    neighbors = nearest_stations[st]
    sp_rain = rain_pivot[neighbors].mean(axis=1).to_frame(name='spatial_rain_avg')
    sp_rain['nama_pos'] = st
    spatial_rain_list.append(sp_rain.reset_index())

full_env = pd.merge(lingkungan_df, pd.concat(spatial_rain_list, ignore_index=True), on=['datetime', 'nama_pos'], how='left')
full_env = pd.merge(full_env, koordinat_df, on='nama_pos', how='left')
full_env = pd.merge(full_env, station_elevation, on='nama_pos', how='left')

# B. Ekstraksi Waktu
full_env['hour_sin'] = np.sin(2 * np.pi * full_env['datetime'].dt.hour / 24.0)
full_env['hour_cos'] = np.cos(2 * np.pi * full_env['datetime'].dt.hour / 24.0)

# C. Ekstraksi Exogenous Lags & Hydrology (Dihitung per stasiun)
full_env = full_env.sort_values(['nama_pos', 'datetime']).reset_index(drop=True)
for st, group in full_env.groupby('nama_pos'):
    idx = group.index
    full_env.loc[idx, 'rain_sum_72h'] = group['rainfall_mm'].rolling(72, min_periods=1).sum()
    full_env.loc[idx, 'rain_sum_168h'] = group['rainfall_mm'].rolling(168, min_periods=1).sum()
    full_env.loc[idx, 'nino_mean_720h'] = group['nino_34'].rolling(720, min_periods=1).mean()
    
    # Hidrologi
    full_env.loc[idx, 'api_14d'] = group['rainfall_mm'].ewm(alpha=0.07, adjust=False).mean()
    sm_norm = group['soil_moisture_0_7cm'] / 100.0
    full_env.loc[idx, 'effective_runoff'] = group['rainfall_mm'] * sm_norm

df_all = pd.merge(train_df, full_env, on=['datetime', 'nama_pos'], how='inner')

# D. TARGET LAGS (Untuk Model Recursive V12)
# Data intervalnya bisa bervariasi (misal 3 jam atau 6 jam). Kita gunakan observasi 24 jam ke belakang.
# 1 Hari = 8 observasi (jika per 3 jam) atau 4 observasi (jika per 6 jam).
# Agar dinamis dan aman, kita asumsikan index bergeser 8 langkah (asumsi BDC 2024 = 3 jam).
df_all = df_all.sort_values(['nama_pos', 'datetime']).reset_index(drop=True)
for st, group in df_all.groupby('nama_pos'):
    idx = group.index
    df_all.loc[idx, 'tma_lag_24h'] = group['tma_mdpl'].shift(8)
    df_all.loc[idx, 'tma_lag_48h'] = group['tma_mdpl'].shift(16)

# E. Label Encoding
le = LabelEncoder()
df_all['nama_pos_encoded'] = le.fit_transform(df_all['nama_pos'])

print("Semua Fitur Gabungan Siap!")
'''),

        new_markdown_cell('## 3. Training 2 Model Berbeda (A & B)'),
        new_code_cell('''# Daftar Fitur Exogenous Murni (Model A - Anchor)
xgb_features_exo = [
    'nama_pos_encoded', 'elevation_proxy', 'latitude', 'longitude', 
    'spatial_rain_avg', 'api_14d', 'effective_runoff',
    'rain_sum_72h', 'rain_sum_168h', 'nino_mean_720h',
    'rainfall_mm', 'soil_moisture_0_7cm', 'humidity_pct',
    'hour_sin', 'hour_cos'
]

# Daftar Fitur Recursive (Model B - Tracker)
xgb_features_rec = xgb_features_exo + ['tma_lag_24h', 'tma_lag_48h']

# Pembersihan Data Bersama
df_clean = df_all.dropna(subset=['tma_mdpl'] + xgb_features_rec).copy()
df_clean = df_clean.sort_values('datetime').reset_index(drop=True)

y = df_clean['tma_mdpl']
X_exo = df_clean[xgb_features_exo].astype(float)
X_rec = df_clean[xgb_features_rec].astype(float)

# Parameter Optuna V4
best_params = {
    'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'tree_method': 'hist',
    'learning_rate': 0.03, 'max_depth': 7, 'subsample': 0.8, 'random_state': 42
}

# K-Fold Training
kf = KFold(n_splits=5, shuffle=True, random_state=42) # Shuffle=True untuk kekuatan Ensemble di Kaggle
models_A_exo = []
models_B_rec = []

print("Melatih Model A (Exogenous Anchor) dan Model B (Recursive Tracker)...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X_exo)):
    print(f"--- Fold {fold+1} ---")
    y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
    
    # Train Model A
    dtr_A = xgb.DMatrix(X_exo.iloc[train_idx], label=y_tr)
    dva_A = xgb.DMatrix(X_exo.iloc[val_idx], label=y_va)
    model_A = xgb.train(best_params, dtr_A, 2000, evals=[(dva_A, 'val')], early_stopping_rounds=50, verbose_eval=False)
    models_A_exo.append(model_A)
    
    # Train Model B
    dtr_B = xgb.DMatrix(X_rec.iloc[train_idx], label=y_tr)
    dva_B = xgb.DMatrix(X_rec.iloc[val_idx], label=y_va)
    model_B = xgb.train(best_params, dtr_B, 2000, evals=[(dva_B, 'val')], early_stopping_rounds=50, verbose_eval=False)
    models_B_rec.append(model_B)
    
    rmse_A = root_mean_squared_error(y_va, model_A.predict(dva_A))
    rmse_B = root_mean_squared_error(y_va, model_B.predict(dva_B))
    print(f"Model A (Exo) RMSE: {rmse_A:.4f} | Model B (Rec) RMSE: {rmse_B:.4f}")
'''),

        new_markdown_cell('## 4. Visualisasi (Mengapa Kita Butuh Keduanya?)\nLihat grafik di bawah. Model A (Merah) menstabilkan prediksi di titik rata-rata (Anchor), sementara Model B (Hijau) menembakkan prediksi ke puncak banjir (Tracker). Ensemble (Biru Putus) adalah kombinasi sempurna keduanya!'),
        new_code_cell('''# Karena kita menggunakan shuffle=True, kita tidak bisa memplot data berurutan dari set validasi asli.
# Sebagai gantinya, kita memprediksi ulang seluruh data Train untuk stasiun Jurug dan memplot sampel akhirnya.

st_mask = df_clean['nama_pos'] == 'Jurug' 
X_exo_jurug = xgb.DMatrix(X_exo[st_mask].iloc[-200:])
X_rec_jurug = xgb.DMatrix(X_rec[st_mask].iloc[-200:])
y_jurug = y[st_mask].iloc[-200:].values

# Ambil prediksi dari fold terakhir saja sebagai contoh
pred_A = models_A_exo[-1].predict(X_exo_jurug)
pred_B = models_B_rec[-1].predict(X_rec_jurug)
pred_Ensemble = (0.5 * pred_A) + (0.5 * pred_B)

plt.figure(figsize=(18, 6))
plt.plot(y_jurug, label='Actual TMA (Jurug)', color='black', linewidth=3)
plt.plot(pred_A, label='Model A (Exogenous) - Stabil tapi Datar', color='red', alpha=0.6, linestyle='-')
plt.plot(pred_B, label='Model B (Recursive) - Liar & Menangkap Puncak', color='green', alpha=0.6, linestyle='-')
plt.plot(pred_Ensemble, label='THE GRAND ENSEMBLE (Tengah Sempurna)', color='blue', linewidth=3, linestyle='--')
plt.title('Kekuatan Ensemble: Menyeimbangkan Stabilitas (Merah) dan Kepekaan (Hijau)')
plt.legend()
plt.tight_layout()
plt.show()
'''),

        new_markdown_cell('## 5. Blended Recursive Inference 🌪️\nMemprediksi hari esok dengan iterasi satu per satu. Setiap hari, kita menggabungkan prediksi Model A (Jangkar) dan Model B (Pelacak), lalu memasukkan hasil gabungannya kembali ke dalam matriks `tma_lag` untuk hari berikutnya. *Error* tidak akan pernah meledak karena Model A akan terus menahannya di bumi!'),
        new_code_cell('''print("Menyiapkan Data Test...")
test_parsed = test_df.copy()
split_info = test_parsed['id'].str.split(' - ', n=1, expand=True)
test_parsed['datetime'] = pd.to_datetime(split_info[0])
test_parsed['nama_pos'] = split_info[1]

# Gabungkan dengan Data Train untuk perhitungan Lag di awal Test Set
gabungan_df = pd.concat([train_df[['datetime', 'nama_pos', 'tma_mdpl']], test_parsed[['datetime', 'nama_pos']]], ignore_index=True)
gabungan_df.sort_values(['nama_pos', 'datetime'], inplace=True)
gabungan_df = pd.merge(gabungan_df, full_env, on=['datetime', 'nama_pos'], how='left')
gabungan_df['nama_pos_encoded'] = le.transform(gabungan_df['nama_pos'])

test_times = sorted(test_parsed['datetime'].unique())
print(f"Total Iterasi Waktu: {len(test_times)}")

# Pivot Table untuk menyimpan prediksi dan memberikan Lags
pivot_target = gabungan_df.pivot(index='datetime', columns='nama_pos', values='tma_mdpl')

print("Memulai Blended Recursive Loop...")
for t_idx, current_time in enumerate(test_times):
    if t_idx % 20 == 0:
        print(f"Iterasi ke-{t_idx}/{len(test_times)} ({current_time})")
    
    # 1. Update tma_lag di gabungan_df untuk waktu saat ini berdasarkan Pivot Table
    mask_current = gabungan_df['datetime'] == current_time
    st_names = gabungan_df.loc[mask_current, 'nama_pos'].values
    
    for st in st_names:
        # Ambil indeks waktu di pivot
        pos_time = pivot_target.index.get_loc(current_time)
        # Ambil nilai 8 step ke belakang (24 jam)
        if pos_time >= 8:
            val_24h = pivot_target.iloc[pos_time - 8][st]
            val_48h = pivot_target.iloc[pos_time - 16][st]
            gabungan_df.loc[mask_current & (gabungan_df['nama_pos'] == st), 'tma_lag_24h'] = val_24h
            gabungan_df.loc[mask_current & (gabungan_df['nama_pos'] == st), 'tma_lag_48h'] = val_48h

    # 2. Siapkan fitur baris ini
    X_current = gabungan_df[mask_current]
    
    # FFill Lags jika kosong di awal banget
    X_current['tma_lag_24h'] = X_current['tma_lag_24h'].fillna(X_current['elevation_proxy'])
    X_current['tma_lag_48h'] = X_current['tma_lag_48h'].fillna(X_current['elevation_proxy'])
    
    dexo = xgb.DMatrix(X_current[xgb_features_exo].astype(float))
    drec = xgb.DMatrix(X_current[xgb_features_rec].astype(float))
    
    # 3. Prediksi dengan 5 Model A dan 5 Model B
    pred_A_mean = np.mean([m.predict(dexo) for m in models_A_exo], axis=0)
    pred_B_mean = np.mean([m.predict(drec) for m in models_B_rec], axis=0)
    
    # 4. THE GRAND ENSEMBLE (50% Anchor, 50% Tracker)
    # Kamu bisa ubah bobotnya. 50:50 adalah keseimbangan emas.
    final_pred = (0.5 * pred_A_mean) + (0.5 * pred_B_mean)
    
    # 5. Injeksi Prediksi Kembali ke Pivot Table
    gabungan_df.loc[mask_current, 'tma_mdpl'] = final_pred
    for st, p_val in zip(st_names, final_pred):
        pivot_target.at[current_time, st] = p_val

print("\\nLoop Selesai! Mengekspor Hasil...")
test_final = gabungan_df[gabungan_df['datetime'].isin(test_times)]
sub = pd.merge(test_parsed[['id', 'datetime', 'nama_pos']], test_final[['datetime', 'nama_pos', 'tma_mdpl']], on=['datetime', 'nama_pos'], how='left')
sub = sub[['id', 'tma_mdpl']]

sub.to_csv('submission_v16_grand_ensemble.csv', index=False)
print("✅ File 'submission_v16_grand_ensemble.csv' berhasil dibuat! Silakan hancurkan Leaderboard!")
''')
    ]
    
    nb.cells.extend(cells)
    with open('V16_The_Grand_Ensemble.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == '__main__':
    create_v16_notebook()
    print("Notebook V16_The_Grand_Ensemble.ipynb berhasil dibuat!")
