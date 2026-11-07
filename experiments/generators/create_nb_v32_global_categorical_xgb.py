import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

def create_v32_notebook():
    nb = new_notebook()
    
    cells = [
        new_markdown_cell('# 🌍 V32 NATIVE GLOBAL CATEGORICAL XGBOOST 🌍\\nIni adalah arsitektur untuk menghancurkan batas 1.610. Kita beralih dari 30 model lokal yang kekurangan data, menjadi **1 Model Global** yang dilatih menggunakan 27.000 baris observasi sekaligus.\\n\\n**Apa yang baru?**\\n1. **Native Categorical:** Kita mengaktifkan fitur `enable_categorical=True` milik XGBoost modern. Mesin kini paham bahwa `nama_pos` adalah lokasi geografi murni, persis seperti algoritma CatBoost.\\n2. **Target Z-Score:** Karena 30 sungai punya ketinggian daratan (mdpl) yang berbeda, kita menyamaratakannya menggunakan Z-Score agar bisa ditebak oleh 1 model global.\\n3. **33 Fitur Super:** Seluruh 27 fitur orisinal V26 Anda dipertahankan, ditambah 6 fitur *Deep Hydrology* (Tanah 4 lapis, MJO, ENSO).'),
        
        new_markdown_cell('## 1. Persiapan Environment & Data'),
        new_code_cell('''!pip install -q xgboost scikit-learn optuna
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

print("Data berhasil dimuat!")
'''),

        new_markdown_cell('## 2. Ekstraksi V26 Base + Deep Hydrology (100% Utuh)'),
        new_code_cell('''print("Mengekstraksi 33 Fitur (V26 Asli + Sains Hidrologi)...")
lingkungan_df = lingkungan_df.sort_values(['nama_pos', 'datetime']).reset_index(drop=True)

def build_deep_hydrology_features(env_df):
    agg_df = env_df[['datetime', 'nama_pos']].copy()
    grouped = env_df.groupby('nama_pos')
    
    # 📌 1. MULTI-LAYER SOIL SATURATION (Deep Hydrology)
    soil_cols = ['soil_moisture_0_7cm', 'soil_moisture_7_28cm', 'soil_moisture_28_100cm', 'soil_moisture_100_255cm']
    if all(c in env_df.columns for c in soil_cols):
        env_df['total_soil_moisture'] = env_df[soil_cols].mean(axis=1)
        agg_df['total_soil_moisture'] = env_df['total_soil_moisture']
        agg_df['soil_sat_mean_6h'] = grouped['total_soil_moisture'].transform(lambda x: x.rolling(6, min_periods=1).mean())
        agg_df['soil_sat_mean_72h'] = grouped['total_soil_moisture'].transform(lambda x: x.rolling(72, min_periods=1).mean())
        
        # 📌 2. EFFECTIVE RUNOFF (Deep Hydrology)
        if 'rainfall_mm' in env_df.columns:
            max_soil = env_df['total_soil_moisture'].max()
            sat_ratio = env_df['total_soil_moisture'] / max_soil
            env_df['effective_runoff'] = env_df['rainfall_mm'] * sat_ratio
            agg_df['effective_runoff'] = env_df['effective_runoff']
            agg_df['eff_runoff_sum_24h'] = grouped['effective_runoff'].transform(lambda x: x.rolling(24, min_periods=1).sum())
    
    # 📌 3. MJO & ENSO MACRO-CLIMATE (Deep Hydrology + V26 Asli)
    if 'mjo_amplitude' in env_df.columns and 'nino_34' in env_df.columns:
        env_df['mjo_enso_power'] = env_df['mjo_amplitude'] * env_df['nino_34']
        agg_df['mjo_enso_power'] = env_df['mjo_enso_power']
        agg_df['mjo_enso_mean_720h'] = grouped['mjo_enso_power'].transform(lambda x: x.rolling(720, min_periods=1).mean())
        
    for c in ['nino_34', 'mjo_phase']:
        if c in env_df.columns:
            agg_df[f'{c}_mean_720h'] = grouped[c].transform(lambda x: x.rolling(720, min_periods=1).mean())
        
    # 📌 4. SOLAR RADIATION (Deep Hydrology)
    if 'solar_radiation_mj_m2' in env_df.columns:
        agg_df['solar_rad_mean_24h'] = grouped['solar_radiation_mj_m2'].transform(lambda x: x.rolling(24, min_periods=1).mean())
        
    # 📌 5. V26 ROLLING SUM KLASIK (Hujan & Cuaca Dasar)
    if 'rainfall_mm' in env_df.columns:
        agg_df['rain_sum_6h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(6, min_periods=1).sum())
        agg_df['rain_sum_72h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(72, min_periods=1).sum())
        agg_df['rain_sum_168h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(168, min_periods=1).sum())
        agg_df['rain_sum_360h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(360, min_periods=1).sum())
        agg_df['rain_sum_720h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(720, min_periods=1).sum())
        
    for c in ['temperature_c', 'wind_speed_kmh', 'surface_pressure_hpa', 'humidity_pct']:
        if c in env_df.columns:
            agg_df[f'{c}_mean_6h'] = grouped[c].transform(lambda x: x.rolling(6, min_periods=1).mean())
        
    return agg_df

env_feat = build_deep_hydrology_features(lingkungan_df)
'''),

        new_markdown_cell('## 3. Injeksi Kategori Stasiun & Memori V26'),
        new_code_cell('''hydro_mapping = {
    "Ketonggo": {"UPLAND_SKM": 21.5, "DIS_AV_CMS": 0.658, "ORD_STRA": 1},
    "Sekayu": {"UPLAND_SKM": 800.9, "DIS_AV_CMS": 30.149, "ORD_STRA": 4},
    "Cepu": {"UPLAND_SKM": 10926.8, "DIS_AV_CMS": 352.11, "ORD_STRA": 6},
    "Karanggeneng": {"UPLAND_SKM": 15087.4, "DIS_AV_CMS": 442.477, "ORD_STRA": 6},
    "Bojonegoro - Kali Kethek": {"UPLAND_SKM": 12778.1, "DIS_AV_CMS": 387.983, "ORD_STRA": 6},
    "Badegan": {"UPLAND_SKM": 244.3, "DIS_AV_CMS": 8.624, "ORD_STRA": 4},
    "Lorog": {"UPLAND_SKM": 47.6, "DIS_AV_CMS": 1.777, "ORD_STRA": 3},
    "Wonogiri Dam": {"UPLAND_SKM": 1344.7, "DIS_AV_CMS": 47.568, "ORD_STRA": 4},
    "Kajangan": {"UPLAND_SKM": 5492.4, "DIS_AV_CMS": 185.525, "ORD_STRA": 5},
    "Boboh Kali Lamong": {"UPLAND_SKM": 519.7, "DIS_AV_CMS": 16.171, "ORD_STRA": 4},
    "Kali Anyar - Kreteg Abang": {"UPLAND_SKM": 426.7, "DIS_AV_CMS": 15.949, "ORD_STRA": 3},
    "Colo Weir": {"UPLAND_SKM": 1569.1, "DIS_AV_CMS": 54.787, "ORD_STRA": 4},
    "Ngadipiro": {"UPLAND_SKM": 406.6, "DIS_AV_CMS": 14.145, "ORD_STRA": 3},
    "Bengkelolor": {"UPLAND_SKM": 4.9, "DIS_AV_CMS": 0.15, "ORD_STRA": 1},
    "Ngrembang": {"UPLAND_SKM": 5.7, "DIS_AV_CMS": 0.21, "ORD_STRA": 1},
    "Karangnongko": {"UPLAND_SKM": 10032.6, "DIS_AV_CMS": 330.184, "ORD_STRA": 6},
    "Kali Pepe - Tugu Boto": {"UPLAND_SKM": 24.9, "DIS_AV_CMS": 0.896, "ORD_STRA": 2},
    "Kedungupit": {"UPLAND_SKM": 22.8, "DIS_AV_CMS": 0.771, "ORD_STRA": 1},
    "Brangkal": {"UPLAND_SKM": 801.6, "DIS_AV_CMS": 17.096, "ORD_STRA": 4},
    "Kali Pepe - PTPN": {"UPLAND_SKM": 7.0, "DIS_AV_CMS": 0.245, "ORD_STRA": 1},
    "Babat": {"UPLAND_SKM": 14342.2, "DIS_AV_CMS": 428.018, "ORD_STRA": 6},
    "Napel": {"UPLAND_SKM": 9838.8, "DIS_AV_CMS": 329.639, "ORD_STRA": 6},
    "Peren": {"UPLAND_SKM": 205.3, "DIS_AV_CMS": 6.934, "ORD_STRA": 3},
    "Jurug": {"UPLAND_SKM": 11.3, "DIS_AV_CMS": 0.387, "ORD_STRA": 1},
    "Jarum": {"UPLAND_SKM": 638.1, "DIS_AV_CMS": 23.14, "ORD_STRA": 4},
    "Gunungsari": {"UPLAND_SKM": 593.4, "DIS_AV_CMS": 21.482, "ORD_STRA": 4},
    "Sumberrejo": {"UPLAND_SKM": 31.1, "DIS_AV_CMS": 0.771, "ORD_STRA": 2},
    "Floodway Bridge C": {"UPLAND_SKM": 8.1, "DIS_AV_CMS": 0.2, "ORD_STRA": 1},
    "Arjowinangun - Pacitan": {"UPLAND_SKM": 642.9, "DIS_AV_CMS": 23.231, "ORD_STRA": 4},
    "Serenan": {"UPLAND_SKM": 2477.8, "DIS_AV_CMS": 86.578, "ORD_STRA": 5}
}

def prepare_global_dataset(base_df, env_feat, historical_tma_df):
    df = base_df.copy()
    
    # 📌 NATIVE CATEGORICAL FEATURE UNTUK XGBOOST
    df['nama_pos_cat'] = df['nama_pos'].astype('category')
    
    # 📌 INJEKSI HYDRORIVERS (STATIC BASIN FEATURES)
    df['upland_skm'] = df['nama_pos'].apply(lambda x: hydro_mapping.get(x, {}).get('UPLAND_SKM', 0))
    df['dis_av_cms'] = df['nama_pos'].apply(lambda x: hydro_mapping.get(x, {}).get('DIS_AV_CMS', 0))
    df['ord_stra'] = df['nama_pos'].apply(lambda x: hydro_mapping.get(x, {}).get('ORD_STRA', 1))
    
    
    # 📌 V26 CYCLICAL WAKTU
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['dayofyear'] = df['datetime'].dt.dayofyear
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12.0)
    df['hour_sin_12h'] = np.sin(2 * np.pi * df['hour']/12.0)
    df['hour_cos_12h'] = np.cos(2 * np.pi * df['hour']/12.0)
    
    df = pd.merge(df, env_feat, on=['datetime', 'nama_pos'], how='left')
    
    # 📌 V26 JANGKAR MEMORI
    df['datetime_minus_1_year'] = df['datetime'] - pd.DateOffset(years=1)
    historical_map = historical_tma_df.set_index(['nama_pos', 'datetime'])['tma_mdpl'].to_dict()
    df['tma_lag_365d'] = df.apply(lambda row: historical_map.get((row['nama_pos'], row['datetime_minus_1_year']), np.nan), axis=1)
    
    last_known_map = historical_tma_df.dropna(subset=['tma_mdpl']).sort_values('datetime').groupby('nama_pos').tail(1).set_index('nama_pos')['tma_mdpl'].to_dict()
    df['last_known_tma'] = df['nama_pos'].map(last_known_map)
    df = df.drop(columns=['datetime_minus_1_year'])
    
    return df

print("Menyusun Train Set (Global 27.000 baris)...")
historical_tma = train_df[['datetime', 'nama_pos', 'tma_mdpl']].copy()
train_full = prepare_global_dataset(train_df, env_feat, historical_tma)

print("Menyusun Test Set (Global)...")
test_parsed = test_df.copy()
split_info = test_parsed['id'].str.split(' - ', n=1, expand=True)
test_parsed['datetime'] = pd.to_datetime(split_info[0])
test_parsed['nama_pos'] = split_info[1]
test_full = prepare_global_dataset(test_parsed, env_feat, historical_tma)

# Fillna Lags untuk Train dan Test
for st in train_full['nama_pos'].unique():
    mask_tr = train_full['nama_pos'] == st
    train_full.loc[mask_tr, 'tma_lag_365d'] = train_full.loc[mask_tr, 'tma_lag_365d'].fillna(train_full.loc[mask_tr, 'last_known_tma'])
    mask_te = test_full['nama_pos'] == st
    test_full.loc[mask_te, 'tma_lag_365d'] = test_full.loc[mask_te, 'tma_lag_365d'].fillna(test_full.loc[mask_te, 'last_known_tma'])

train_full = train_full.dropna(subset=['tma_mdpl']).reset_index(drop=True)
drop_cols = ['datetime', 'tma_mdpl', 'id', 'nama_pos'] # nama_pos_cat is used instead for XGBoost
features = [c for c in train_full.columns if c not in drop_cols]
print(f"Total Fitur Global: {len(features)} (Termasuk 3 Fitur HydroRIVERS & Kategori)")
'''),

        new_markdown_cell('## 4. Robust Target Scaling & Anomaly Downweighting\\nKarena kita menyatukan 30 sungai dengan ketinggian tanah (mdpl) yang sangat berbeda, kita wajib mengkonversi target `tma_mdpl` menjadi `z-score`.\\nKita juga memberi beban pinalti pada data error (spike sensor) agar tidak merusak model global.'),
        new_code_cell('''print("Menghitung Statistik Normalisasi (Robust Scaling) per Pos...")
station_stats = {}
for pos in train_full['nama_pos'].unique():
    tma_vals = train_full.loc[train_full['nama_pos'] == pos, 'tma_mdpl']
    med = tma_vals.median()
    q75, q25 = np.percentile(tma_vals, [75, 25])
    iqr = q75 - q25
    scale = max(iqr / 1.349, 0.05) # Sesuai standar robust statistics
    station_stats[pos] = {'median': med, 'scale': scale}

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
    rolling_med = df_sorted.groupby('nama_pos')['z_tma'].transform(lambda x: x.rolling(window, center=True, min_periods=1).median())
    
    anom_count = 0
    for z, r_med in zip(df_sorted['z_tma'], rolling_med):
        if pd.isna(z):
            weights.append(1.0)
            continue
        if abs(z - r_med) > threshold:
            weights.append(0.05) # Hukuman untuk noise (abaikan)
            anom_count += 1
        else:
            weights.append(1.0)
            
    df_sorted['sample_weight'] = weights
    print(f"-> Terdeteksi {anom_count} baris data spike (Bobot disunat menjadi 0.05)")
    return df_sorted.sort_index()

train_full = calculate_sample_weights(train_full)
'''),

        new_markdown_cell('## 5. Optuna Global K-Fold (The Cheat Code is Back)\\nModel Global yang diawasi Optuna dan K-Fold(Shuffle). Model ini melihat 27.000 baris acak dari seluruh musim dan seluruh stasiun secara paralel!'),
        new_code_cell('''# Matikan log Optuna agar bersih
optuna.logging.set_verbosity(optuna.logging.ERROR)
N_TRIALS = 30 # Lebihkan sedikit karena kita cuma punya 1 model global (bukan 30 model)

print(f"Memulai Optuna Global XGBoost ({N_TRIALS} Trials)...\\n")

X = train_full[features]
y = train_full['z_tma']
w = train_full['sample_weight']

def objective(trial):
    params = {
        'objective': 'reg:pseudohubererror', # Paling aman untuk target Z-Score dan pencilan
        'eval_metric': 'mae', 
        'tree_method': 'hist',
        'device': 'cuda', 
        'enable_categorical': True, # 📌 KUNCI UTAMA (XGBOOST SEKARANG SETARA CATBOOST)
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 5, 12), # Depth besar tidak overfit karena data banyak
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'n_estimators': 800,
        'random_state': 42
    }
    
    # Pakai K-Fold 3 Splits saja untuk kecepatan Optuna
    kf_optuna = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    
    for tr_idx, va_idx in kf_optuna.split(X):
        X_tr, y_tr, w_tr = X.iloc[tr_idx], y.iloc[tr_idx], w.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]
        
        model = xgb.XGBRegressor(**params, early_stopping_rounds=30)
        model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_va, y_va)], verbose=False)
        scores.append(root_mean_squared_error(y_va, model.predict(X_va)))
        
    return np.mean(scores)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=N_TRIALS)

best_params = study.best_params
best_params.update({
    'objective': 'reg:pseudohubererror', 'eval_metric': 'mae', 'tree_method': 'hist', 
    'device': 'cuda', 'enable_categorical': True, 'n_estimators': 1500, 'random_state': 42
})

print(f"👑 Best Global Parameters: {best_params}")

# ================= PELATIHAN FINAL DENGAN 5-FOLD (SHUFFLE) =================
print("\\nMelatih 5-Fold Final Global Model...")
kf_final = KFold(n_splits=5, shuffle=True, random_state=42)
final_models = []
val_scores_meter = []

for tr_idx, va_idx in kf_final.split(X):
    X_tr, y_tr, w_tr = X.iloc[tr_idx], y.iloc[tr_idx], w.iloc[tr_idx]
    X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]
    
    model = xgb.XGBRegressor(**best_params, early_stopping_rounds=50)
    model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_va, y_va)], verbose=False)
    
    # Evaluasi menggunakan Meter Asli (Bukan Z-Score)
    z_preds = model.predict(X_va)
    tma_actuals = train_full.iloc[va_idx]['tma_mdpl'].values
    pos_names = train_full.iloc[va_idx]['nama_pos'].values
    tma_preds = []
    
    for i, pos in enumerate(pos_names):
        st = station_stats[pos]
        tma_preds.append((z_preds[i] * st['scale']) + st['median'])
        
    fold_rmse = root_mean_squared_error(tma_actuals, tma_preds)
    val_scores_meter.append(fold_rmse)
    final_models.append(model)
    print(f"   -> Fold RMSE (Meter): {fold_rmse:.4f}")

print(f"\\n🏆 GRAND CV RMSE (Global XGBoost): {np.mean(val_scores_meter):.4f}")
'''),

        new_markdown_cell('## 6. Prediksi Test Set (Inverse Transform)'),
        new_code_cell('''print("Mengeksekusi Prediksi pada Test Set...")
test_full['tma_mdpl'] = np.nan
test_full = test_full.fillna(method='ffill').fillna(0) # Safety fill

X_test = test_full[features]
z_test_preds = np.zeros(len(X_test))

for m in final_models:
    z_test_preds += m.predict(X_test) / len(final_models)

test_full['z_tma'] = z_test_preds

# Mengembalikan nilai ke METER (Inverse Scaling)
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
sub.to_csv('submission_v32_global_categorical_xgb.csv', index=False)
print("\\n🚀 File 'submission_v32_global_categorical_xgb.csv' BERHASIL DIBUAT dengan kecerdasan Global Native XGBoost!")
''')
    ]
    
    nb.cells.extend(cells)
    with open('V32_Global_Categorical_XGB.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == '__main__':
    create_v32_notebook()
    print("Notebook V32_Global_Categorical_XGB.ipynb berhasil dibuat!")
