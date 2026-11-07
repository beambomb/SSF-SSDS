import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

def create_v31_notebook():
    nb = new_notebook()
    
    cells = [
        new_markdown_cell('# 🌊 V31 THE DEEP HYDROLOGY (Optuna + V26 Full Features) 🌊\\nSesuai permintaan Anda, kita kembalikan kekuatan pencarian Hyperparameter **Optuna** per stasiun.\\nKita juga memastikan **SEMUA 27 Fitur orisinal V26** kembali masuk, ditambah dengan racikan *Deep Hydrology* (Tanah 4 lapis, Effective Runoff, MJO x ENSO, Radiasi Matahari) sehingga total fitur kini menembus **30+ Fitur super!**'),
        
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

        new_markdown_cell('## 2. Ekstraksi Harta Karun (Deep Hydrology + V26 Full Features)'),
        new_code_cell('''print("Mengekstraksi Fitur Sains Iklim & Hidrologi Dalam...")
lingkungan_df = lingkungan_df.sort_values(['nama_pos', 'datetime']).reset_index(drop=True)

def build_deep_hydrology_features(env_df):
    agg_df = env_df[['datetime', 'nama_pos']].copy()
    grouped = env_df.groupby('nama_pos')
    
    # 📌 1. MULTI-LAYER SOIL SATURATION (0 - 2.5 Meter)
    soil_cols = ['soil_moisture_0_7cm', 'soil_moisture_7_28cm', 'soil_moisture_28_100cm', 'soil_moisture_100_255cm']
    if all(c in env_df.columns for c in soil_cols):
        env_df['total_soil_moisture'] = env_df[soil_cols].mean(axis=1)
        agg_df['total_soil_moisture'] = env_df['total_soil_moisture']
        agg_df['soil_sat_mean_6h'] = grouped['total_soil_moisture'].transform(lambda x: x.rolling(6, min_periods=1).mean())
        agg_df['soil_sat_mean_72h'] = grouped['total_soil_moisture'].transform(lambda x: x.rolling(72, min_periods=1).mean())
        
        # 📌 2. EFFECTIVE RUNOFF (Banjir Kilat jika tanah jenuh)
        if 'rainfall_mm' in env_df.columns:
            max_soil = env_df['total_soil_moisture'].max()
            sat_ratio = env_df['total_soil_moisture'] / max_soil
            env_df['effective_runoff'] = env_df['rainfall_mm'] * sat_ratio
            agg_df['effective_runoff'] = env_df['effective_runoff']
            agg_df['eff_runoff_sum_24h'] = grouped['effective_runoff'].transform(lambda x: x.rolling(24, min_periods=1).sum())
    
    # 📌 3. MJO & ENSO MACRO-CLIMATE DYNAMICS
    if 'mjo_amplitude' in env_df.columns and 'nino_34' in env_df.columns:
        env_df['mjo_enso_power'] = env_df['mjo_amplitude'] * env_df['nino_34']
        agg_df['mjo_enso_power'] = env_df['mjo_enso_power']
        agg_df['mjo_enso_mean_720h'] = grouped['mjo_enso_power'].transform(lambda x: x.rolling(720, min_periods=1).mean())
        
    # Mengembalikan fitur V26 yang hilang: nino_34 murni dan mjo_phase murni
    for c in ['nino_34', 'mjo_phase']:
        if c in env_df.columns:
            agg_df[f'{c}_mean_720h'] = grouped[c].transform(lambda x: x.rolling(720, min_periods=1).mean())
        
    # 📌 4. SOLAR RADIATION (Energi Evaporasi Murni)
    if 'solar_radiation_mj_m2' in env_df.columns:
        agg_df['solar_rad_mean_24h'] = grouped['solar_radiation_mj_m2'].transform(lambda x: x.rolling(24, min_periods=1).mean())
        
    # 📌 5. V26 ROLLING SUM KLASIK
    if 'rainfall_mm' in env_df.columns:
        agg_df['rain_sum_6h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(6, min_periods=1).sum())
        agg_df['rain_sum_72h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(72, min_periods=1).sum())
        agg_df['rain_sum_168h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(168, min_periods=1).sum())
        agg_df['rain_sum_360h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(360, min_periods=1).sum())
        agg_df['rain_sum_720h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(720, min_periods=1).sum())
        
    # Standard weather V26
    for c in ['temperature_c', 'wind_speed_kmh', 'surface_pressure_hpa', 'humidity_pct']:
        if c in env_df.columns:
            agg_df[f'{c}_mean_6h'] = grouped[c].transform(lambda x: x.rolling(6, min_periods=1).mean())
        
    return agg_df

env_feat = build_deep_hydrology_features(lingkungan_df)
'''),

        new_markdown_cell('## 3. Injeksi Jangkar Memori V26 & Cyclical Time'),
        new_code_cell('''def prepare_local_dataset(base_df, env_feat, historical_tma_df):
    df = base_df.copy()
    
    # 📌 Mengembalikan Cyclical Features V26 yang Hilang
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

print("Menyusun Train Set (V31)...")
historical_tma = train_df[['datetime', 'nama_pos', 'tma_mdpl']].copy()
train_full = prepare_local_dataset(train_df, env_feat, historical_tma)

print("Menyusun Test Set (V31)...")
test_parsed = test_df.copy()
split_info = test_parsed['id'].str.split(' - ', n=1, expand=True)
test_parsed['datetime'] = pd.to_datetime(split_info[0])
test_parsed['nama_pos'] = split_info[1]
test_full = prepare_local_dataset(test_parsed, env_feat, historical_tma)

# Fillna Lags
for st in train_full['nama_pos'].unique():
    mask_tr = train_full['nama_pos'] == st
    train_full.loc[mask_tr, 'tma_lag_365d'] = train_full.loc[mask_tr, 'tma_lag_365d'].fillna(train_full.loc[mask_tr, 'last_known_tma'])
    
    mask_te = test_full['nama_pos'] == st
    test_full.loc[mask_te, 'tma_lag_365d'] = test_full.loc[mask_te, 'tma_lag_365d'].fillna(test_full.loc[mask_te, 'last_known_tma'])

train_full = train_full.dropna(subset=['tma_mdpl']).reset_index(drop=True)

drop_cols = ['datetime', 'nama_pos', 'tma_mdpl', 'id']
features = [c for c in train_full.columns if c not in drop_cols]
print(f"Total Fitur per Pos: {len(features)} (Sudah termasuk V26 + Deep Hydrology)")
'''),

        new_markdown_cell('## 4. Optuna K-Fold Shuffle (The Kaggle Cheat Code)\\nMencari Hyperparameter spesifik untuk masing-masing 30 stasiun agar mendapatkan *fitting* sempurna!'),
        new_code_cell('''stations = train_full['nama_pos'].unique()
station_models = {}
station_scores = {}

# Matikan log Optuna agar tidak memenuhi layar
optuna.logging.set_verbosity(optuna.logging.ERROR)
N_TRIALS = 15

print(f"Memulai Optuna (Tweedie) untuk 30 Stasiun (Total {30 * N_TRIALS} Trials)...\\n")

for st in stations:
    X_st = train_full[train_full['nama_pos'] == st][features]
    
    # 📌 KUNCI BUGFIX: Tweedie tidak bisa menerima nilai negatif (Kasus Ketonggo)
    y_st = train_full[train_full['nama_pos'] == st]['tma_mdpl'].clip(lower=0) 
    
    def objective(trial):
        params = {
            'objective': 'reg:tweedie', 
            'tweedie_variance_power': trial.suggest_float('tweedie_variance_power', 1.1, 1.9),
            'eval_metric': 'rmse', # Gunakan RMSE standar untuk Optuna agar aman
            'tree_method': 'hist',
            'device': 'cuda', # Gunakan GPU untuk mempercepat
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'n_estimators': 800,
            'random_state': 42
        }
        
        # Validasi 80-20 khusus untuk Optuna
        split_idx = int(len(X_st) * 0.8)
        X_tr, y_tr = X_st.iloc[:split_idx], y_st.iloc[:split_idx]
        X_va, y_va = X_st.iloc[split_idx:], y_st.iloc[split_idx:]
        
        model = xgb.XGBRegressor(**params, early_stopping_rounds=30)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        
        preds = model.predict(X_va)
        return root_mean_squared_error(y_va, preds)

    # Jalankan Optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS)
    
    # Parameter Terbaik
    best_params = study.best_params
    best_params['objective'] = 'reg:tweedie'
    best_params['eval_metric'] = 'rmse'
    best_params['tree_method'] = 'hist'
    best_params['device'] = 'cuda'
    best_params['n_estimators'] = 800
    best_params['random_state'] = 42

    # Latih ulang dengan 5-Fold menggunakan parameter terbaik
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    models = []
    scores = []
    
    for train_idx, val_idx in kf.split(X_st, y_st):
        X_tr, y_tr = X_st.iloc[train_idx], y_st.iloc[train_idx]
        X_va, y_va = X_st.iloc[val_idx], y_st.iloc[val_idx]
        
        model = xgb.XGBRegressor(**best_params, early_stopping_rounds=50)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        
        val_preds = model.predict(X_va)
        scores.append(root_mean_squared_error(y_va, val_preds))
        models.append(model)
        
    avg_rmse = np.mean(scores)
    station_models[st] = models
    station_scores[st] = avg_rmse
    
    print(f"[{st}] - RMSE: {avg_rmse:.4f}")

global_rmse = np.mean(list(station_scores.values()))
print(f"\\n🏆 GRAND AVERAGE RMSE ACROSS 30 STATIONS: {global_rmse:.4f}")
'''),

        new_markdown_cell('## 5. Visualisasi & Pembuatan Submission'),
        new_code_cell('''TARGET_STATION = "Jurug"

fig, ax = plt.subplots(1, 2, figsize=(22, 8))

# 1. Feature Importance
feat_imp = np.mean([m.feature_importances_ for m in station_models[TARGET_STATION]], axis=0)
imp_df = pd.DataFrame({'Features': features, 'Importance score': feat_imp})
imp_df = imp_df.sort_values(by='Importance score', ascending=False).head(20)

import seaborn as sns
sns.barplot(x='Importance score', y='Features', data=imp_df, ax=ax[0], palette='viridis')
ax[0].set_title(f'Top 20 Feature Importance [{TARGET_STATION}]')

# 2. Actual vs Pred
X_target = train_full[train_full['nama_pos'] == TARGET_STATION][features]
y_target = train_full[train_full['nama_pos'] == TARGET_STATION]['tma_mdpl']

val_preds = np.zeros(len(X_target))
for m in station_models[TARGET_STATION]:
    val_preds += m.predict(X_target) / 5

sub_actual = y_target.iloc[-200:].values
sub_pred = val_preds[-200:]
ax[1].plot(sub_actual, label='Actual TMA', color='blue', alpha=0.7, linewidth=2)
ax[1].plot(sub_pred, label='Predicted (Deep Hydrology XGB)', color='red', alpha=0.7, linewidth=2)
ax[1].set_title(f'Actual vs Predicted - 200 Hari Terakhir [{TARGET_STATION}]')
ax[1].legend()

plt.tight_layout()
plt.show()

# ================= BAGIAN TEST SET =================
print("Memprediksi Test Set...")
test_full['tma_mdpl'] = np.nan
test_full = test_full.fillna(method='ffill').fillna(0) # Safety fill untuk fitur NaN

for st in stations:
    idx = test_full['nama_pos'] == st
    if idx.sum() == 0: continue
        
    X_test_st = test_full.loc[idx, features]
    
    st_preds = np.zeros(len(X_test_st))
    for m in station_models[st]:
        st_preds += m.predict(X_test_st) / 5
        
    test_full.loc[idx, 'tma_mdpl'] = st_preds

sub_pred = test_full[['id', 'tma_mdpl']].copy()
sub = pd.merge(sample_sub[['id']], sub_pred, on='id', how='left')

sub['tma_mdpl'] = sub['tma_mdpl'].fillna(method='ffill').fillna(method='bfill')

display(sub.head())
sub.to_csv('submission_v31_deep_hydrology_optuna.csv', index=False)
print("\\n🚀 File 'submission_v31_deep_hydrology_optuna.csv' BERHASIL DIBUAT!")
''')
    ]
    
    nb.cells.extend(cells)
    with open('V31_Deep_Hydrology_XGB.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == '__main__':
    create_v31_notebook()
    print("Notebook V31_Deep_Hydrology_XGB.ipynb berhasil di-rebuild dengan Optuna!")
