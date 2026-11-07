import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

def create_v14_notebook():
    nb = new_notebook()
    
    cells = [
        new_markdown_cell('# ⚔️ V14 THE MODEL BATTLE: XGB vs LGBM vs CAT vs RF\nSaatnya mempertarungkan 4 raksasa *Machine Learning*! Kita akan melatih keempatnya secara bersamaan dan mencetak 4 file *submission* berbeda agar kamu bisa melihat siapa algoritma terbaik di Kaggle!'),
        
        new_markdown_cell('## 1. Persiapan Environment & Pemuatan Data'),
        new_code_cell('''!pip install -q xgboost lightgbm catboost scikit-learn scipy
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as plt_sns
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from scipy.spatial.distance import cdist

plt.style.use('seaborn-v0_8-whitegrid')

# Auto-Detect Path (Kaggle / Colab / Local)
if not 'path' in locals():
    if os.path.exists('/kaggle/input'):
        dataset_folders = [f for f in os.listdir('/kaggle/input') if os.path.isdir(f'/kaggle/input/{f}')]
        path = f'/kaggle/input/{dataset_folders[0]}/' if dataset_folders else '/kaggle/input/'
    elif os.path.exists('/content'):
        path = '/content/drive/MyDrive/ssf-ssds/'
    else:
        path = './'
print(f"Path dataset aktif: {path}")

# Load Data Utama
train_df = pd.read_csv(path + 'train.csv', parse_dates=['datetime'])
lingkungan_df = pd.read_csv(path + 'data_pendukung/data_lingkungan.csv', parse_dates=['datetime'])
koordinat_df = pd.read_csv(path + 'data_pendukung/koordinat_pos.csv')

print(f"Train Shape: {train_df.shape}")
print(f"Lingkungan Shape: {lingkungan_df.shape}")
print(f"Koordinat Shape: {koordinat_df.shape}")
'''),

        new_markdown_cell('## 2. Spatial Topology (Pengganti HydroRIVERS & GWNet)\nKita menghitung matriks jarak antar stasiun berdasarkan Latitude & Longitude. Lalu, untuk setiap stasiun, kita ambil curah hujan dari 3 stasiun terdekatnya. Ini secara otomatis mengajarkan XGBoost arah aliran air dan hujan regional!'),
        new_code_cell('''# 1. Menghitung Jarak Euclidean antar Stasiun
coords = koordinat_df[['latitude', 'longitude']].values
dist_matrix = cdist(coords, coords)

station_names = koordinat_df['nama_pos'].values
nearest_stations = {}

print("Memetakan 3 Stasiun Tetangga Terdekat (Topologi Spasial):")
for i, st in enumerate(station_names):
    # argsort mengurutkan jarak dari terdekat ke terjauh
    # Index 0 adalah dirinya sendiri (jarak 0), jadi ambil index 1 sampai 3
    nearest_idx = np.argsort(dist_matrix[i])[1:4]
    nearest_stations[st] = station_names[nearest_idx]
    if i < 3: # Print contoh
        print(f"-> {st} bertetangga dengan: {list(nearest_stations[st])}")

# 2. Membuat Fitur Hujan Regional (Spatial Rain)
rain_pivot = lingkungan_df.pivot(index='datetime', columns='nama_pos', values='rainfall_mm')

spatial_rain_list = []
for st in station_names:
    neighbors = nearest_stations[st]
    # Rata-rata hujan di 3 stasiun tetangga
    sp_rain = rain_pivot[neighbors].mean(axis=1).to_frame(name='spatial_rain_avg')
    sp_rain['nama_pos'] = st
    spatial_rain_list.append(sp_rain.reset_index())

spatial_rain_df = pd.concat(spatial_rain_list, ignore_index=True)

# 3. Gabungkan Spatial Rain ke data Lingkungan
full_env = pd.merge(lingkungan_df, spatial_rain_df, on=['datetime', 'nama_pos'], how='left')

# 4. Tambahkan Latitude dan Longitude Asli
full_env = pd.merge(full_env, koordinat_df, on='nama_pos', how='left')

print(f"\\nFitur Spatial Lags berhasil ditambahkan! Shape: {full_env.shape}")
'''),

        new_markdown_cell('## 3. Massive Feature Engineering\nMenggabungkan seluruh "Harta Karun" cuaca: Mikro, Makro, Tanah, Siklus Waktu, *Rolling*, dan *Lag* (sebagai memori). **NO TARGET CLIPPING!** Data target dibiarkan murni agar model bisa menebak puncak banjir ekstrem.'),
        new_code_cell('''# A. Iklim Mikro Instan & Siklus Waktu
full_env['dew_point_spread'] = full_env['temperature_c'] - full_env['dew_point_c']
full_env['hour_sin'] = np.sin(2 * np.pi * full_env['datetime'].dt.hour / 24.0)
full_env['hour_cos'] = np.cos(2 * np.pi * full_env['datetime'].dt.hour / 24.0)
full_env['month_sin'] = np.sin(2 * np.pi * full_env['datetime'].dt.month / 12.0)
full_env['month_cos'] = np.cos(2 * np.pi * full_env['datetime'].dt.month / 12.0)
full_env['wind_sin'] = np.sin(np.radians(full_env['wind_direction_deg']))
full_env['wind_cos'] = np.cos(np.radians(full_env['wind_direction_deg']))

# B. Rolling & Lags (Memori Hulu-Hilir Eksogen) - DIHITUNG PER STASIUN
print("Menghitung Rolling & Lags Cuaca... (Aman untuk Test Set)")
for st, group in full_env.groupby('nama_pos'):
    idx = group.index
    
    # 1. Rolling Akumulasi Hujan (Kelembaban sistem)
    full_env.loc[idx, 'rain_roll_12h'] = group['rainfall_mm'].rolling(12, min_periods=1).sum()
    full_env.loc[idx, 'rain_roll_24h'] = group['rainfall_mm'].rolling(24, min_periods=1).sum()
    full_env.loc[idx, 'rain_roll_72h'] = group['rainfall_mm'].rolling(72, min_periods=1).sum()
    full_env.loc[idx, 'rain_roll_168h'] = group['rainfall_mm'].rolling(168, min_periods=1).sum()
    
    # 2. Lags Hujan Lokal (Memori Air Turun)
    # 8 step = 24 jam (karena 1 step = 3 jam, asumsi data harian jika 1 step = 1 jam)
    # Dataset BDC biasanya per jam atau per 6 jam. Mari kita pakai shift langsung (asumsi berurutan)
    full_env.loc[idx, 'rain_lag_1'] = group['rainfall_mm'].shift(1)
    full_env.loc[idx, 'rain_lag_2'] = group['rainfall_mm'].shift(2)
    full_env.loc[idx, 'rain_lag_3'] = group['rainfall_mm'].shift(3)
    full_env.loc[idx, 'rain_lag_4'] = group['rainfall_mm'].shift(4)
    full_env.loc[idx, 'rain_lag_8'] = group['rainfall_mm'].shift(8) 
    
    # 3. Lags Spatial Hujan (Tetangga)
    full_env.loc[idx, 'spatial_rain_lag_1'] = group['spatial_rain_avg'].shift(1)
    full_env.loc[idx, 'spatial_rain_lag_2'] = group['spatial_rain_avg'].shift(2)
    full_env.loc[idx, 'spatial_rain_lag_4'] = group['spatial_rain_avg'].shift(4)
    
    # 4. Profil Tanah
    full_env.loc[idx, 'soil_diff_0_7cm_24h'] = group['soil_moisture_0_7cm'].diff(8)

# Gabungkan dengan target (TMA) - MURNI TANPA CLIPPING!
df_all = pd.merge(train_df, full_env, on=['datetime', 'nama_pos'], how='inner')

# C. One-Hot Encoding (OHE) Stasiun
ohe_pos = pd.get_dummies(df_all['nama_pos'], prefix='pos', drop_first=False)
df_all = pd.concat([df_all, ohe_pos], axis=1)

print(f"Dataset Tabular Murni Siap! Shape: {df_all.shape}")
'''),

        new_markdown_cell('## 4. BATTLE ARENA (Training 4 Algoritma)\nMelatih XGBoost, LightGBM, CatBoost, dan Random Forest secara bersamaan!'),
        new_code_cell('''print("Menyiapkan Porsi Makanan (~80 Fitur)...")
# A. Daftarkan Fitur
xgb_features = [
    'latitude', 'longitude',
    'spatial_rain_avg', 'spatial_rain_lag_1', 'spatial_rain_lag_2', 'spatial_rain_lag_4',
    'rainfall_mm', 'humidity_pct', 'cloud_cover_pct', 'solar_radiation_mj_m2',
    'soil_moisture_0_7cm', 'soil_moisture_7_28cm', 'soil_moisture_28_100cm', 'soil_moisture_100_255cm',
    'nino_34', 'rmm1', 'rmm2', 'mjo_amplitude', 'mjo_active',
    'month_sin', 'month_cos', 'hour_sin', 'hour_cos',
    'rain_roll_12h', 'rain_roll_24h', 'rain_roll_72h', 'rain_roll_168h',
    'rain_lag_1', 'rain_lag_2', 'rain_lag_3', 'rain_lag_4', 'rain_lag_8',
    'soil_diff_0_7cm_24h'
]

pos_cols = [c for c in df_all.columns if c.startswith('pos_')]
xgb_features.extend(pos_cols)

# B. Pembersihan Baris (Missing Values dari Lags)
df_clean = df_all.dropna(subset=['tma_mdpl'] + xgb_features)
df_clean = df_clean.sort_values('datetime').reset_index(drop=True)

X = df_clean[xgb_features].astype(float)
y = df_clean['tma_mdpl']

print(f"Data Bersih Siap Latih: X={X.shape}, y={y.shape}\\n")

# C. KAMUS MODEL
models_dict = {
    'XGBoost': xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=8, tree_method='hist', random_state=42),
    'LightGBM': lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, max_depth=8, random_state=42, verbose=-1),
    'CatBoost': CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=8, random_seed=42, verbose=0),
    'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42) # RF cukup 100 agar tidak lama
}

trained_models = {name: [] for name in models_dict.keys()}
rmse_scores = {name: [] for name in models_dict.keys()}

# D. K-Fold Training Loop
kf = KFold(n_splits=3, shuffle=False)

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"--- Fold {fold + 1} ---")
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    for name, model_template in models_dict.items():
        # Clone model template
        from sklearn.base import clone
        model = clone(model_template)
        
        # Fit model (berbeda sintaks eval_set untuk RF)
        if name in ['XGBoost', 'LightGBM']:
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
        elif name == 'CatBoost':
            model.fit(X_tr, y_tr, eval_set=(X_va, y_va))
        else:
            # Random Forest tidak punya eval_set bawaan di sklearn
            model.fit(X_tr, y_tr)
            
        val_rmse = root_mean_squared_error(y_va, model.predict(X_va))
        rmse_scores[name].append(val_rmse)
        trained_models[name].append(model)
        
print("\\n🏆 === HASIL PERTANDINGAN (Rata-rata RMSE Validation) ===")
for name in models_dict.keys():
    print(f"{name:<15}: {np.mean(rmse_scores[name]):.4f}")
'''),

        new_markdown_cell('## 5. Visualisasi Karakter Model\nMari kita lihat bagaimana gaya tebakan keempat model ini. Apakah ada yang bergerigi (*overfit*)? Apakah ada yang sangat mulus?'),
        new_code_cell('''import matplotlib.pyplot as plt

plt.figure(figsize=(20, 8))

# Sampel Babat - Fold Validasi Akhir
st_mask = df_clean['pos_Babat'] == 1.0 
X_babat = df_clean.loc[st_mask, xgb_features].astype(float)
y_babat = df_clean.loc[st_mask, 'tma_mdpl']

if len(X_babat) > 300:
    X_sub = X_babat.iloc[-300:]
    y_sub = y_babat.iloc[-300:].values
    
    plt.plot(y_sub, label='Actual TMA (Babat)', color='black', linewidth=3)
    
    colors = {'XGBoost': 'red', 'LightGBM': 'blue', 'CatBoost': 'purple', 'RandomForest': 'green'}
    
    for name in models_dict.keys():
        # Ambil model fold terakhir
        model = trained_models[name][-1]
        pred_sub = model.predict(X_sub)
        plt.plot(pred_sub, label=f'{name} Predict', color=colors[name], alpha=0.7, linestyle='--', linewidth=2)
        
    plt.title('Pertarungan 4 Model (Actual vs Predicted - Validation Set Babat)')
    plt.legend()

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
'''),

        new_markdown_cell('## 6. Inference Instan (Cetak 4 CSV!)\nSesuai janjiku, alih-alih melempar hasil Ensemble, saya akan memisahkan prediksi keempat model ini ke dalam 4 file CSV yang berbeda. Silakan kamu submit keempatnya ke Kaggle dan lihat siapa yang mendapat emas!'),
        new_code_cell('''print("Memulai proses Inference Test Data (Mencetak 4 CSV)...")

# 1. Muat Data Test
test_df = pd.read_csv(path + 'test.csv')
test_df['datetime'] = pd.to_datetime(test_df['id'].str.split(' - ', n=1).str[0])
test_df['nama_pos'] = test_df['id'].str.split(' - ', n=1).str[1]

# 2. Gabungkan Train & Test untuk perhitungan Lags yang mulus
gabungan_df = pd.concat([train_df[['datetime', 'nama_pos']], test_df[['datetime', 'nama_pos']]], ignore_index=True)
gabungan_df.sort_values(['nama_pos', 'datetime'], inplace=True)

# Merge dengan full_env yang sudah memiliki ALL features
gabungan_df = pd.merge(gabungan_df, full_env, on=['datetime', 'nama_pos'], how='left')
ohe_pos_test = pd.get_dummies(gabungan_df['nama_pos'], prefix='pos', drop_first=False)
gabungan_df = pd.concat([gabungan_df, ohe_pos_test], axis=1)

# 3. Filter khusus data waktu Test
test_times = sorted(test_df['datetime'].unique())
test_final = gabungan_df[gabungan_df['datetime'].isin(test_times)].copy()
test_features = test_final[xgb_features].astype(float)

# 4. Prediksi & Cetak CSV untuk masing-masing model
for name in models_dict.keys():
    print(f"Menembak Prediksi dengan {name}...")
    
    # Rata-ratakan prediksi dari seluruh K-Fold model tersebut
    fold_preds = []
    for model in trained_models[name]:
        fold_preds.append(model.predict(test_features))
    
    avg_pred = np.mean(fold_preds, axis=0)
    
    test_final['tma_mdpl'] = avg_pred
    submission = pd.merge(test_df[['id', 'datetime', 'nama_pos']], test_final[['datetime', 'nama_pos', 'tma_mdpl']], on=['datetime', 'nama_pos'], how='left')
    submission = submission[['id', 'tma_mdpl']]
    
    filename = f'submission_v14_{name.lower()}.csv'
    submission.to_csv(filename, index=False)
    print(f" -> {filename} berhasil dicetak!")

print("\\nSELESAI! Silakan bereksperimen dengan 4 file submission_v14 tersebut di Kaggle!")
''')
    ]
    
    nb.cells.extend(cells)
    with open('V14_Model_Battle.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == '__main__':
    create_v14_notebook()
    print("Notebook V14_Model_Battle.ipynb berhasil dibuat!")
