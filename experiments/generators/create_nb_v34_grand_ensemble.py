import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

def create_v34_notebook():
    nb = new_notebook()
    
    cells = [
        new_markdown_cell('# 🔥 V34: THE GRAND ENSEMBLE (CatBoost/LGBM Killer) 🔥\\n\\nSelamat datang di arsitektur tingkat *Grandmaster*! Notebook ini dibangun berdasarkan usulan Anda untuk menggabungkan dua mesin prediktif (XGBoost + LightGBM), membubuhi 80+ fitur canggih (seperti *Exponential Moving Average* untuk serapan tanah), dan meracik tebakan dari **Top-3 Optuna Parameter** sekaligus.\\n\\nNotebook ini sudah dilengkapi pengaturan Google Colab untuk kemudahan Anda.'),
        
        new_markdown_cell('## 1. Google Colab & Kaggle Setup (Administrasi Drive)'),
        new_code_cell('''!pip install -q xgboost lightgbm scikit-learn optuna

import os, warnings, gc
warnings.filterwarnings('ignore')

import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
import xgboost as xgb, lightgbm as lgb, optuna
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold

plt.style.use('seaborn-v0_8-whitegrid')

# ----------------- ADMINISTRASI GOOGLE DRIVE / KAGGLE -----------------
path = './'
if os.path.exists('/content'):
    print("Mendeteksi Google Colab. Melakukan proses mount Google Drive...")
    from google.colab import drive
    drive.mount('/content/drive')
    # Sesuaikan path di bawah ini dengan lokasi folder Anda di Google Drive
    colab_path = '/content/drive/MyDrive/SSF-SSDS/'
    if os.path.exists(colab_path):
        path = colab_path
    else:
        print(f"Peringatan: Folder {colab_path} tidak ditemukan, pastikan data ada di sana.")
elif os.path.exists('/kaggle/input'):
    print("Mendeteksi environment Kaggle...")
    dataset_folders = [f for f in os.listdir('/kaggle/input') if os.path.isdir(f'/kaggle/input/{f}')]
    path = f'/kaggle/input/{dataset_folders[0]}/' if dataset_folders else '/kaggle/input/'

print(f"Path data yang digunakan: {path}\\n")

train_df = pd.read_csv(path + 'train.csv', parse_dates=['datetime'])
test_df = pd.read_csv(path + 'test.csv')
lingkungan_df = pd.read_csv(path + 'data_pendukung/data_lingkungan.csv', parse_dates=['datetime'])
sample_sub = pd.read_csv(path + 'sample_submission.csv')
koordinat_df = pd.read_csv(path + 'data_pendukung/koordinat_pos.csv')

print("Data berhasil dimuat!")
'''),

        new_markdown_cell('## 2. Pembangunan 80+ Fitur (EWM & Spatial Graph)'),
        new_code_cell('''print("Mengekstraksi Fitur Sains Hidrologi & EWM (Antecedent Precipitation)...")
lingkungan_df = lingkungan_df.sort_values(['nama_pos', 'datetime']).reset_index(drop=True)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0 # Radius Bumi (km)
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    a = np.sin(dLat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dLon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

# Matriks Jarak Geografis Antar Stasiun
coord_dict = koordinat_df.set_index('nama_pos').to_dict('index')
dist_matrix = {}
for pos1 in coord_dict:
    dist_matrix[pos1] = {}
    for pos2 in coord_dict:
        if pos1 == pos2:
            dist_matrix[pos1][pos2] = 0
        else:
            dist_matrix[pos1][pos2] = haversine(coord_dict[pos1]['latitude'], coord_dict[pos1]['longitude'],
                                                coord_dict[pos2]['latitude'], coord_dict[pos2]['longitude'])

def build_advanced_features(env_df):
    agg_df = env_df[['datetime', 'nama_pos']].copy()
    grouped = env_df.groupby('nama_pos')
    
    # 📌 1. MULTI-LAYER SOIL & EFFECTIVE RUNOFF
    soil_cols = ['soil_moisture_0_7cm', 'soil_moisture_7_28cm', 'soil_moisture_28_100cm', 'soil_moisture_100_255cm']
    if all(c in env_df.columns for c in soil_cols):
        env_df['total_soil_moisture'] = env_df[soil_cols].mean(axis=1)
        agg_df['total_soil_moisture'] = env_df['total_soil_moisture']
        agg_df['soil_sat_mean_24h'] = grouped['total_soil_moisture'].transform(lambda x: x.rolling(24, min_periods=1).mean())
        
        if 'rainfall_mm' in env_df.columns:
            max_soil = env_df['total_soil_moisture'].max()
            sat_ratio = env_df['total_soil_moisture'] / max_soil
            env_df['effective_runoff'] = env_df['rainfall_mm'] * sat_ratio
            agg_df['effective_runoff'] = env_df['effective_runoff']
            agg_df['eff_runoff_sum_72h'] = grouped['effective_runoff'].transform(lambda x: x.rolling(72, min_periods=1).sum())
    
    # 📌 2. ANTECEDENT PRECIPITATION INDEX (API) - EWM (Kunci dari 84 Fitur)
    # Menyimulasikan penyerapan hujan ke dalam tanah dengan Exponential Decay
    if 'rainfall_mm' in env_df.columns:
        # Hujan Harian, Mingguan
        agg_df['rain_sum_6h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(6, min_periods=1).sum())
        agg_df['rain_sum_24h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(24, min_periods=1).sum())
        agg_df['rain_sum_72h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(72, min_periods=1).sum())
        agg_df['rain_sum_168h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(168, min_periods=1).sum())
        
        # Hujan Terserap (EWM)
        agg_df['rain_ewm_12h'] = grouped['rainfall_mm'].transform(lambda x: x.ewm(halflife=12, ignore_na=True).mean())
        agg_df['rain_ewm_48h'] = grouped['rainfall_mm'].transform(lambda x: x.ewm(halflife=48, ignore_na=True).mean())
        agg_df['rain_ewm_168h'] = grouped['rainfall_mm'].transform(lambda x: x.ewm(halflife=168, ignore_na=True).mean())
        
    # 📌 3. CUACA LOKAL & MAKRO
    if 'mjo_amplitude' in env_df.columns and 'nino_34' in env_df.columns:
        agg_df['mjo_enso_power'] = env_df['mjo_amplitude'] * env_df['nino_34']
    for c in ['temperature_c', 'humidity_pct']:
        if c in env_df.columns:
            agg_df[f'{c}_mean_24h'] = grouped[c].transform(lambda x: x.rolling(24, min_periods=1).mean())
            
    return agg_df

env_feat = build_advanced_features(lingkungan_df)

# 📌 4. SPATIAL GRAPH AGGREGATION (Hujan Jarak Jauh / Kiriman)
print("Menghitung Hujan Spasial / Jaringan Sungai (Spatial Decay)...")
env_feat['spatial_rain_sum_24h'] = 0.0

# Pivot data curah hujan untuk kemudahan perhitungan spasial per timestamp
if 'rain_sum_24h' in env_feat.columns:
    rain_pivot = env_feat.pivot(index='datetime', columns='nama_pos', values='rain_sum_24h').fillna(0)
    
    spatial_features = []
    for pos in rain_pivot.columns:
        # Hitung bobot kedekatan geografis exp(-jarak / 75km) sesuai paper
        weights = pd.Series({p2: np.exp(-dist_matrix[pos][p2]/75.0) for p2 in rain_pivot.columns})
        weights[pos] = 0 # Jangan masukkan diri sendiri, karena kita murni mengukur "Kiriman Air" dari luar
        
        # Hitung total sum
        weighted_rain = (rain_pivot * weights).sum(axis=1)
        temp_df = pd.DataFrame({'datetime': rain_pivot.index, 'nama_pos': pos, 'spatial_rain_sum_24h': weighted_rain.values})
        spatial_features.append(temp_df)
        
    spatial_df = pd.concat(spatial_features, ignore_index=True)
    env_feat = env_feat.drop(columns=['spatial_rain_sum_24h'], errors='ignore')
    env_feat = pd.merge(env_feat, spatial_df, on=['datetime', 'nama_pos'], how='left')
'''),

        new_markdown_cell('## 3. Dataset Global (Tanpa Z-Score, Target Murni Meter)'),
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
    
    # 📌 Native Categorical untuk XGBoost & LightGBM
    df['nama_pos_cat'] = df['nama_pos'].astype('category')
    
    # 📌 Injeksi HydroRIVERS
    df['upland_skm'] = df['nama_pos'].apply(lambda x: hydro_mapping.get(x, {}).get('UPLAND_SKM', 0))
    df['dis_av_cms'] = df['nama_pos'].apply(lambda x: hydro_mapping.get(x, {}).get('DIS_AV_CMS', 0))
    df['ord_stra'] = df['nama_pos'].apply(lambda x: hydro_mapping.get(x, {}).get('ORD_STRA', 1))
    
    # 📌 Fitur Waktu Kalender
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['dayofyear'] = df['datetime'].dt.dayofyear
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12.0)
    
    df = pd.merge(df, env_feat, on=['datetime', 'nama_pos'], how='left')
    
    # 📌 Memori Autoregresif 1 Tahun
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

# Fallback untuk lag yang kosong
for st in train_full['nama_pos'].unique():
    mask_tr = train_full['nama_pos'] == st
    train_full.loc[mask_tr, 'tma_lag_365d'] = train_full.loc[mask_tr, 'tma_lag_365d'].fillna(train_full.loc[mask_tr, 'last_known_tma'])
    mask_te = test_full['nama_pos'] == st
    test_full.loc[mask_te, 'tma_lag_365d'] = test_full.loc[mask_te, 'tma_lag_365d'].fillna(test_full.loc[mask_te, 'last_known_tma'])

train_full = train_full.dropna(subset=['tma_mdpl']).reset_index(drop=True)
drop_cols = ['datetime', 'tma_mdpl', 'id', 'nama_pos']
features = [c for c in train_full.columns if c not in drop_cols]
print(f"Total Fitur Final: {len(features)}")
'''),

        new_markdown_cell('## 4. Optuna Top-K Ensemble (Juara 1, 2, & 3 XGBoost)\\nDi sini kita tidak sembarangan menebak. Optuna berjalan 20 kali, dan kita **SIMPAN 3 PARAMETER TERBAIK** untuk menjamin variasi (Parameter Ensembling).'),
        new_code_cell('''optuna.logging.set_verbosity(optuna.logging.ERROR)
N_TRIALS_XGB = 20

X = train_full[features]
y = train_full['tma_mdpl'].clip(lower=0) 

def objective_xgb(trial):
    params = {
        'objective': 'reg:tweedie', 
        'tweedie_variance_power': trial.suggest_float('tweedie_variance_power', 1.1, 1.9),
        'eval_metric': 'rmse',
        'tree_method': 'hist',
        'device': 'cuda', # Ganti 'cpu' jika jalan di lokal tanpa GPU
        'enable_categorical': True, 
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 5, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'n_estimators': 300,
        'random_state': 42
    }
    
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    
    for tr_idx, va_idx in kf.split(X):
        model = xgb.XGBRegressor(**params, early_stopping_rounds=15)
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx], eval_set=[(X.iloc[va_idx], y.iloc[va_idx])], verbose=False)
        scores.append(root_mean_squared_error(y.iloc[va_idx], model.predict(X.iloc[va_idx])))
        
    return np.mean(scores)

print(f"Mencari Top-K XGBoost Parameters ({N_TRIALS_XGB} Trials)...")
study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(objective_xgb, n_trials=N_TRIALS_XGB)

# Mengambil 3 percobaan terbaik
top_trials_xgb = sorted(study_xgb.trials, key=lambda t: t.value)[:3]
print("\\nTop 3 XGBoost Parameter Ditemukan!")
for i, t in enumerate(top_trials_xgb):
    print(f"Rank {i+1}: RMSE {t.value:.4f}")

# Pelatihan 3 Model XGBoost Final
print("\\nMelatih 3 Model XGBoost Berbeda...")
xgb_models = []
kf_final = KFold(n_splits=5, shuffle=True, random_state=42)

for i, trial in enumerate(top_trials_xgb):
    params = trial.params.copy()
    params.update({'objective': 'reg:tweedie', 'eval_metric': 'rmse', 'tree_method': 'hist', 'device': 'cuda', 'enable_categorical': True, 'n_estimators': 1500, 'random_state': 42+i})
    
    # Latih model ini pada 1 fold (atau Full data, namun KFold lebih validasi)
    tr_idx, va_idx = next(kf_final.split(X))
    model = xgb.XGBRegressor(**params, early_stopping_rounds=50)
    model.fit(X.iloc[tr_idx], y.iloc[tr_idx], eval_set=[(X.iloc[va_idx], y.iloc[va_idx])], verbose=False)
    
    preds = model.predict(X.iloc[va_idx])
    fold_rmse = root_mean_squared_error(y.iloc[va_idx], preds)
    xgb_models.append(model)
    print(f"XGBoost Model {i+1} terlatih. Val RMSE: {fold_rmse:.4f}")
'''),

        new_markdown_cell('## 5. Optuna Top-K Ensemble (Juara 1, 2, & 3 LightGBM)\\nSekarang giliran mesin vertikal (*leaf-wise*) beraksi.'),
        new_code_cell('''N_TRIALS_LGB = 20

def objective_lgb(trial):
    params = {
        'objective': 'tweedie',
        'tweedie_variance_power': trial.suggest_float('tweedie_variance_power', 1.1, 1.9),
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 256),
        'max_depth': trial.suggest_int('max_depth', 5, 12),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': 1,
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
        'verbose': -1,
        'device': 'gpu', # Ganti ke 'cpu' jika Kaggle/Colab error GPU LGBM
        'random_state': 42
    }
    
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    
    for tr_idx, va_idx in kf.split(X):
        dtrain = lgb.Dataset(X.iloc[tr_idx], label=y.iloc[tr_idx], categorical_feature=['nama_pos_cat'])
        dval = lgb.Dataset(X.iloc[va_idx], label=y.iloc[va_idx], reference=dtrain, categorical_feature=['nama_pos_cat'])
        
        # Gunakan callback standard LightGBM
        callbacks = [lgb.early_stopping(15, verbose=False)]
        
        model = lgb.train(params, dtrain, valid_sets=[dval], num_boost_round=300, callbacks=callbacks)
        scores.append(root_mean_squared_error(y.iloc[va_idx], model.predict(X.iloc[va_idx])))
        
    return np.mean(scores)

print(f"Mencari Top-K LightGBM Parameters ({N_TRIALS_LGB} Trials)...")
study_lgb = optuna.create_study(direction='minimize')
study_lgb.optimize(objective_lgb, n_trials=N_TRIALS_LGB)

top_trials_lgb = sorted(study_lgb.trials, key=lambda t: t.value)[:3]
print("\\nTop 3 LightGBM Parameter Ditemukan!")
for i, t in enumerate(top_trials_lgb):
    print(f"Rank {i+1}: RMSE {t.value:.4f}")

# Pelatihan 3 Model LGBM Final
print("\\nMelatih 3 Model LightGBM Berbeda...")
lgb_models = []
kf_final = KFold(n_splits=5, shuffle=True, random_state=42)

for i, trial in enumerate(top_trials_lgb):
    params = trial.params.copy()
    params.update({'objective': 'tweedie', 'metric': 'rmse', 'boosting_type': 'gbdt', 'verbose': -1, 'device': 'gpu', 'random_state': 42+i})
    
    tr_idx, va_idx = next(kf_final.split(X))
    dtrain = lgb.Dataset(X.iloc[tr_idx], label=y.iloc[tr_idx], categorical_feature=['nama_pos_cat'])
    dval = lgb.Dataset(X.iloc[va_idx], label=y.iloc[va_idx], reference=dtrain, categorical_feature=['nama_pos_cat'])
    
    callbacks = [lgb.early_stopping(50, verbose=False)]
    model = lgb.train(params, dtrain, valid_sets=[dval], num_boost_round=1500, callbacks=callbacks)
    
    preds = model.predict(X.iloc[va_idx])
    fold_rmse = root_mean_squared_error(y.iloc[va_idx], preds)
    lgb_models.append(model)
    print(f"LightGBM Model {i+1} terlatih. Val RMSE: {fold_rmse:.4f}")
'''),

        new_markdown_cell('## 6. Prediksi Raksasa (The Grand Mean Ensemble) & Cetak Submission'),
        new_code_cell('''print("Mengeksekusi The Grand Ensemble pada Test Set...")
test_full['tma_mdpl'] = np.nan

# Safety fill untuk fitur numerik
for col in test_full.columns:
    if test_full[col].dtype != 'category':
        test_full[col] = test_full[col].fillna(method='ffill').fillna(0)

X_test = test_full[features]
test_preds = np.zeros(len(X_test))

all_models = xgb_models + lgb_models
weight_per_model = 1.0 / len(all_models)

for idx, m in enumerate(all_models):
    if "XGB" in str(type(m)):
        test_preds += m.predict(X_test) * weight_per_model
        print(f"XGBoost {idx+1} telah menyumbang tebakan.")
    else:
        test_preds += m.predict(X_test) * weight_per_model
        print(f"LightGBM {idx+1} telah menyumbang tebakan.")

test_full['tma_mdpl'] = test_preds

sub_pred = test_full[['id', 'tma_mdpl']].copy()
sub = pd.merge(sample_sub[['id']], sub_pred, on='id', how='left')
sub['tma_mdpl'] = sub['tma_mdpl'].fillna(method='ffill').fillna(method='bfill')

display(sub.head())
sub.to_csv('submission_v34_grand_ensemble.csv', index=False)
print("\\n🚀 File 'submission_v34_grand_ensemble.csv' BERHASIL DIBUAT (Siap Mengguncang Leaderboard)!")
'''),
        
        new_markdown_cell('## 7. Visualisasi Evaluasi (Bagian Terpisah Sesuai Request)\\nLihat "Dapur" dari model Anda, fitur mana yang menurut algoritma paling penting!'),
        new_code_cell('''fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# 1. Plot XGBoost Feature Importance
xgb_importance = xgb_models[0].get_booster().get_score(importance_type='weight')
xgb_imp_df = pd.DataFrame({'Feature': list(xgb_importance.keys()), 'Importance': list(xgb_importance.values())})
xgb_imp_df = xgb_imp_df.sort_values(by='Importance', ascending=False).head(20)

sns.barplot(data=xgb_imp_df, x='Importance', y='Feature', ax=axes[0], palette='viridis')
axes[0].set_title('Top 20 Fitur Terpenting (XGBoost Juara 1)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('F-Score / Weight')

# 2. Plot LightGBM Feature Importance
lgb_importance = lgb_models[0].feature_importance(importance_type='split')
lgb_imp_df = pd.DataFrame({'Feature': features, 'Importance': lgb_importance})
lgb_imp_df = lgb_imp_df.sort_values(by='Importance', ascending=False).head(20)

sns.barplot(data=lgb_imp_df, x='Importance', y='Feature', ax=axes[1], palette='magma')
axes[1].set_title('Top 20 Fitur Terpenting (LightGBM Juara 1)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Split Count')

plt.tight_layout()
plt.show()

# 3. Distribusi Prediksi
plt.figure(figsize=(10, 5))
sns.histplot(sub['tma_mdpl'], bins=100, kde=True, color='blue', alpha=0.6)
plt.title('Distribusi Prediksi Target (The Grand Ensemble)', fontweight='bold')
plt.xlabel('Prediksi TMA (Meter)')
plt.ylabel('Frekuensi')
plt.show()
'''),
        
        new_markdown_cell('## 8. Komparasi Interaktif (V26 vs V34)\\nMari kita bandingkan secara visual hasil tebakan V34 (The Grand Ensemble) dengan V26 (Lokal murni) kesayangan Anda! Pastikan Anda sudah *upload* file `submission_v26.csv` (atau apa pun nama filenya) ke *environment* Colab ini.'),
        new_code_cell('''# Instal ipywidgets jika belum ada
!pip install -q ipywidgets

import ipywidgets as widgets
from IPython.display import display

# GANTI NAMA FILE V26 DI BAWAH INI JIKA BERBEDA
file_v26 = 'submission_v26.csv' 
file_v34 = 'submission_v34_grand_ensemble.csv'

if os.path.exists(file_v26) and os.path.exists(file_v34):
    sub26 = pd.read_csv(file_v26)
    sub34 = pd.read_csv(file_v34)
    
    split_id = sub34['id'].str.split(' - ', n=1, expand=True)
    
    comp_df = pd.DataFrame({
        'datetime': pd.to_datetime(split_id[0]),
        'nama_pos': split_id[1],
        'Prediksi_V26': sub26['tma_mdpl'].values,
        'Prediksi_V34': sub34['tma_mdpl'].values
    })
    
    comp_df = comp_df.sort_values('datetime')
    stasiun_list = sorted(comp_df['nama_pos'].unique())
    
    def plot_comparison(stasiun):
        df_plot = comp_df[comp_df['nama_pos'] == stasiun]
        
        plt.figure(figsize=(16, 6))
        plt.plot(df_plot['datetime'], df_plot['Prediksi_V26'], label='V26 (Lokal murni)', color='red', alpha=0.7, linestyle='--')
        plt.plot(df_plot['datetime'], df_plot['Prediksi_V34'], label='V34 (Grand Ensemble)', color='blue', alpha=0.9, linewidth=2)
        
        plt.title(f'Komparasi Prediksi TMA di Stasiun: {stasiun}', fontweight='bold', fontsize=14)
        plt.xlabel('Tanggal')
        plt.ylabel('Tinggi Muka Air (Meter)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    dropdown = widgets.Dropdown(options=stasiun_list, description='Pilih Stasiun:')
    print("Silakan pilih stasiun dari menu dropdown di bawah:")
    display(widgets.interactive(plot_comparison, stasiun=dropdown))
else:
    print(f"File {file_v26} belum ditemukan! Silakan upload file hasil submission V26 Anda agar grafik bisa ditampilkan.")
''')
    ]
    
    nb.cells.extend(cells)
    with open('V34_Grand_Ensemble_Colab.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == '__main__':
    create_v34_notebook()
    print("Notebook V34_Grand_Ensemble_Colab.ipynb berhasil dibuat!")
