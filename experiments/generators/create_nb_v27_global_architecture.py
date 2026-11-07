import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

def create_v27_notebook():
    nb = new_notebook()
    
    cells = [
        new_markdown_cell('# 🌍 V27 THE GLOBAL HYBRID ARCHITECTURE 🌍\\nSelamat datang di arsitektur tingkat akhir! Di sini kita meniru teknik **God-Tier Engineering** dari dokumen rahasia Anda:\\n1. **Sensor Anomaly Downweighting:** Mengabaikan *spike* palsu akibat sensor korslet (memberi bobot 0.05).\\n2. **Robust Target Scaling:** Menseragamkan elevasi 30 sungai (Z-Score berbasis IQR) agar bisa diproses oleh **1 Model Global**.\\n3. **Temporal Rolling-Origin CV:** Mencegah kebocoran waktu dengan melatih murni masa lalu untuk menebak masa depan.\\n4. **Fisika Air (API & Dew Point):** Mengganti kalkulasi Hujan menjadi Peluruhan Eksponensial (EWM) dan menghitung evaporasi musim kemarau.\\n\\n*Loss Function* berganti menjadi `reg:pseudo_hubererror` karena target $Z$ kini memiliki rentang negatif/positif.'),
        
        new_markdown_cell('## 1. Persiapan Environment & Data'),
        new_code_cell('''!pip install -q xgboost scikit-learn optuna folium
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import optuna
from sklearn.metrics import root_mean_squared_error

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

        new_markdown_cell('## 2. Pemetaan Topologi Spasial (Hulu ke Hilir)'),
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
'''),

        new_markdown_cell('## 3. The "God-Tier" Feature Engineering (API & Anomaly Downweighting)'),
        new_code_cell('''print("Mengekstraksi Fitur Fisika Air (EWM & Dew-Point)...")
lingkungan_df = lingkungan_df.sort_values(['nama_pos', 'datetime']).reset_index(drop=True)

def build_exogenous_features(env_df):
    agg_df = env_df[['datetime', 'nama_pos']].copy()
    grouped = env_df.groupby('nama_pos')
    
    # 📌 1. DEW-POINT DEPRESSION (Rumus Evaporasi Magnus)
    if 'temperature_c' in env_df.columns and 'humidity_pct' in env_df.columns:
        env_df['dew_point'] = env_df['temperature_c'] - ((100 - env_df['humidity_pct']) / 5.0)
        env_df['dew_point_dep'] = env_df['temperature_c'] - env_df['dew_point']
        agg_df['dew_point_dep_mean_6h'] = grouped['dew_point_dep'].transform(lambda x: x.rolling(6, min_periods=1).mean())
        
    # 📌 2. ANTECEDENT PRECIPITATION INDEX (EWM) - Menggantikan Rolling Sum biasa
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
            # Mengikuti arsitektur: indeks makro iklim di-smooth sangat lambat
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

def prepare_global_dataset(base_df, env_feat, upstream_map, is_train=True):
    df = base_df.copy()
    
    # Waktu
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['dayofyear'] = df['datetime'].dt.dayofyear
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12.0)
    
    # Encode Kategori Pos
    df['nama_pos_encoded'] = df['nama_pos'].astype('category').cat.codes
    
    df = pd.merge(df, env_feat, on=['datetime', 'nama_pos'], how='left')
    
    # 📌 4. SUNTIKAN 3 FITUR HULU (UPSTREAM FEATURES)
    print("  -> Menyuntikkan Fitur Aliran Sungai Hulu (Upstream)...")
    upstream_dfs = []
    for pos, upstreams in upstream_map.items():
        if len(upstreams) > 0:
            up_df = df[df['nama_pos'].isin(upstreams)]
            agg = up_df.groupby('datetime')[['api_48h', 'dew_point_dep_mean_6h', 'soil_mean_72h']].mean().reset_index()
        else:
            up_df = df[df['nama_pos'] == pos]
            agg = up_df[['datetime', 'api_48h', 'dew_point_dep_mean_6h', 'soil_mean_72h']].copy()
        
        agg = agg.rename(columns={
            'api_48h': 'upstream_api_48h',
            'dew_point_dep_mean_6h': 'upstream_dew_point_dep',
            'soil_mean_72h': 'upstream_soil_mean_72h'
        })
        agg['nama_pos'] = pos
        upstream_dfs.append(agg)
        
    all_upstream = pd.concat(upstream_dfs, ignore_index=True)
    df = pd.merge(df, all_upstream, on=['datetime', 'nama_pos'], how='left')
    
    return df

print("\\nMenyusun Train Set Global...")
train_full = prepare_global_dataset(train_df, env_feat, upstream_map, is_train=True)

print("Menyusun Test Set Global...")
test_parsed = test_df.copy()
split_info = test_parsed['id'].str.split(' - ', n=1, expand=True)
test_parsed['datetime'] = pd.to_datetime(split_info[0])
test_parsed['nama_pos'] = split_info[1]
test_full = prepare_global_dataset(test_parsed, env_feat, upstream_map, is_train=False)

train_full = train_full.dropna(subset=['tma_mdpl']).reset_index(drop=True)
'''),

        new_markdown_cell('## 4. Robust Scaling & Sensor Anomaly Downweighting\\nDi sini kita menghapus elevasi stasiun (`Z-Score IQR`) dan memberikan bobot `0.05` pada *spike* yang tidak masuk akal!'),
        new_code_cell('''print("Menghitung Statistik Normalisasi (Robust Scaling) per Pos...")
station_stats = {}
for pos in train_full['nama_pos'].unique():
    tma_vals = train_full.loc[train_full['nama_pos'] == pos, 'tma_mdpl']
    med = tma_vals.median()
    q75, q25 = np.percentile(tma_vals, [75, 25])
    iqr = q75 - q25
    # Sesuai architecture: max(IQR / 1.349, 0.05)
    scale = max(iqr / 1.349, 0.05)
    station_stats[pos] = {'median': med, 'scale': scale}

# Terapkan Robust Scaling
def apply_robust_scaling(df, stats):
    z_tma = []
    for idx, row in df.iterrows():
        st = stats[row['nama_pos']]
        z = (row['tma_mdpl'] - st['median']) / st['scale']
        z_tma.append(z)
    return z_tma

train_full['z_tma'] = apply_robust_scaling(train_full, station_stats)

print("Mendeteksi Spike Sensor Palsu (Anomaly Downweighting)...")
def calculate_sample_weights(df, window=5, threshold=3.0):
    weights = []
    df_sorted = df.sort_values(['nama_pos', 'datetime'])
    
    # Hitung rolling median
    rolling_med = df_sorted.groupby('nama_pos')['z_tma'].transform(lambda x: x.rolling(window, center=True, min_periods=1).median())
    
    anom_count = 0
    for z, r_med in zip(df_sorted['z_tma'], rolling_med):
        if pd.isna(z):
            weights.append(1.0)
            continue
        # Jika lonjakan melebihi 3 deviasi IQR murni sekilas
        if abs(z - r_med) > threshold:
            weights.append(0.05) # "Abaikan spike ini"
            anom_count += 1
        else:
            weights.append(1.0)
            
    df_sorted['sample_weight'] = weights
    print(f"-> Terdeteksi {anom_count} baris data spike (Bobot disunat menjadi 0.05)")
    return df_sorted.sort_index()

train_full = calculate_sample_weights(train_full)

drop_cols = ['datetime', 'nama_pos', 'tma_mdpl', 'z_tma', 'id', 'sample_weight']
features = [c for c in train_full.columns if c not in drop_cols]
print(f"Total Fitur Global: {len(features)}")
'''),

        new_markdown_cell('## 5. Optuna XGBoost (Global Model) + Temporal Rolling-Origin CV\\nModel dilatih dengan `reg:pseudo_hubererror` dan 100% Taat Waktu (Anti-Leakage)'),
        new_code_cell('''# Cutoff Waktu Validasi (Sesuai Architecture)
cutoffs = [
    ('2023-09-18', '2023-09-19', '2024-05-18'), # Fold 1
    ('2024-05-18', '2024-05-19', '2025-01-18'), # Fold 2
    ('2025-01-18', '2025-01-19', '2025-09-18')  # Fold 3
]

optuna.logging.set_verbosity(optuna.logging.INFO)
N_TRIALS = 15

print(f"Memulai Optuna Global Huber ({N_TRIALS} Trials)...\\n")

X = train_full[features]
y = train_full['z_tma']
w = train_full['sample_weight']
dt = train_full['datetime']

def objective(trial):
    params = {
        'objective': 'reg:pseudohubererror', # Menggantikan Tweedie karena Z bisa negatif
        'eval_metric': 'mae', 
        'tree_method': 'hist',
        'device': 'cuda', # WAJIB UNTUK MEMPERCEPAT JUTAAN BARIS DI KAGGLE
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 5, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'n_estimators': 1500,
        'random_state': 42
    }
    
    cv_scores = []
    
    # TEMPORAL CV LOOP
    for tr_end, val_start, val_end in cutoffs:
        train_mask = dt <= pd.to_datetime(tr_end)
        val_mask = (dt >= pd.to_datetime(val_start)) & (dt <= pd.to_datetime(val_end))
        
        X_tr, y_tr, w_tr = X[train_mask], y[train_mask], w[train_mask]
        X_va, y_va = X[val_mask], y[val_mask]
        
        model = xgb.XGBRegressor(**params, early_stopping_rounds=30)
        # Injeksi sample weight agar model mengabaikan sensor error
        model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_va, y_va)], verbose=False)
        
        z_preds = model.predict(X_va)
        
        # Evaluasi asli dalam skala METER (Bukan Z-Score)
        tma_actuals = train_full.loc[val_mask, 'tma_mdpl'].values
        pos_names = train_full.loc[val_mask, 'nama_pos'].values
        tma_preds = []
        for i, pos in enumerate(pos_names):
            st = station_stats[pos]
            tma_preds.append((z_preds[i] * st['scale']) + st['median'])
            
        cv_scores.append(root_mean_squared_error(tma_actuals, tma_preds))
        
    return np.mean(cv_scores)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=N_TRIALS)

best_params = study.best_params
best_params['objective'] = 'reg:pseudohubererror'
best_params['eval_metric'] = 'mae'
best_params['tree_method'] = 'hist'
best_params['device'] = 'cuda'
best_params['n_estimators'] = 1500
best_params['random_state'] = 42

print(f"👑 Best Global Parameters: {best_params}")
print(f"👑 Best Validation RMSE (Meter): {study.best_value:.4f}")
'''),

        new_markdown_cell('## 6. Training Ensemble Final (3 Seed) & Prediksi Test Set\\nSama seperti CatBoost architecture, kita menggabungkan 3 Seed berbeda (17, 41, 83) untuk stabilitas ekstra.'),
        new_code_cell('''seeds = [17, 41, 83]
final_models = []

print("\\nMelatih 3 Global Model Final dengan Seed Berbeda...")
for s in seeds:
    bp = best_params.copy()
    bp['random_state'] = s
    
    # Karena kita akan tebak 8 bulan ke depan, kita train pakai 100% data
    model = xgb.XGBRegressor(**bp)
    model.fit(X, y, sample_weight=w, verbose=False)
    final_models.append(model)
    print(f"✅ Model Seed {s} Selesai!")

print("\\nMengeksekusi Prediksi pada Test Set (Inverse Transform)...")
X_test = test_full[features]

z_test_preds = np.zeros(len(X_test))
for m in final_models:
    z_test_preds += m.predict(X_test) / len(seeds)

test_full['z_tma'] = z_test_preds

# Mengembalikan nilai ke METER
final_tma = []
for idx, row in test_full.iterrows():
    pos = row['nama_pos']
    st = station_stats[pos]
    tma = (row['z_tma'] * st['scale']) + st['median']
    final_tma.append(tma)

test_full['tma_mdpl'] = final_tma

sub_pred = test_full[['id', 'tma_mdpl']].copy()
sub = pd.merge(sample_sub[['id']], sub_pred, on='id', how='left')
sub['tma_mdpl'] = sub['tma_mdpl'].fillna(method='ffill').fillna(method='bfill')

display(sub.head())
sub.to_csv('submission_v27_global_architecture.csv', index=False)
print("\\n🚀 File 'submission_v27_global_architecture.csv' berhasil dibuat dengan Arsitektur Global (Huber + Robust Scaling)!")
''')
    ]
    
    nb.cells.extend(cells)
    with open('V27_Global_Architecture.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == '__main__':
    create_v27_notebook()
    print("Notebook V27_Global_Architecture.ipynb berhasil dibuat!")
