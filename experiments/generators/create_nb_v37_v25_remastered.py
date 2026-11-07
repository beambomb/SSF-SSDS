import nbformat as nbf

def create_v37_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # Cell 1: Intro & Data Loading
    cell1 = nbf.v4.new_code_cell('''# ==============================================================================
# V37: V25 REMASTERED (LOCAL TWEEDIE GOAT + EWM + TIMESERIESSPLIT)
# Arsitektur: 30 Independent Local XGBoost Models dengan Tweedie Loss
# Fitur: V25 Basin Features + HydroRIVERS + EWM Physics + 365d Lag Memory
# Validasi: TimeSeriesSplit (No Future Leakage)
# ==============================================================================

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

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

print("Memuat dataset...")
train_df = pd.read_csv(path + 'train.csv', parse_dates=['datetime'])
test_df = pd.read_csv(path + 'test.csv')
lingkungan_df = pd.read_csv(path + 'data_lingkungan.csv', parse_dates=['datetime'])
koordinat_df = pd.read_csv(path + 'data_pendukung/koordinat_pos.csv')
sample_sub = pd.read_csv(path + 'sample_submission.csv')

print("Data berhasil dimuat!")''')
    cells.append(cell1)

    # Cell 2: Feature Engineering
    cell2 = nbf.v4.new_code_cell('''# ==============================================================================
# FEATURE ENGINEERING: LOKAL, BASIN (GLOBAL), EWM, DAN HYDRORIVERS
# ==============================================================================
print("Mengekstraksi Fitur Cuaca Lokal, Global Basin, dan EWM Physics...")
lingkungan_df = lingkungan_df.sort_values(['nama_pos', 'datetime']).reset_index(drop=True)

def build_exogenous_features(env_df):
    agg_df = env_df[['datetime', 'nama_pos']].copy()
    
    # 📌 1. FITUR LOKAL V25 (Rolling Windows) & V37 (EWM)
    grouped = env_df.groupby('nama_pos')
    
    if 'rainfall_mm' in env_df.columns:
        # V25 Original
        agg_df['rain_sum_6h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(6, min_periods=1).sum())
        agg_df['rain_sum_72h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(72, min_periods=1).sum())
        agg_df['rain_sum_168h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(168, min_periods=1).sum())
        agg_df['rain_sum_360h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(360, min_periods=1).sum())
        agg_df['rain_sum_720h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(720, min_periods=1).sum())
        
        # V37 Injection: EWM Physics (Sangat penting untuk hidrologi)
        agg_df['rain_ewm_48h'] = grouped['rainfall_mm'].transform(lambda x: x.ewm(halflife=48, ignore_na=True).mean())
        agg_df['rain_ewm_168h'] = grouped['rainfall_mm'].transform(lambda x: x.ewm(halflife=168, ignore_na=True).mean())
        
    if 'soil_moisture_0_7cm' in env_df.columns:
        agg_df['soil_mean_6h'] = grouped['soil_moisture_0_7cm'].transform(lambda x: x.rolling(6, min_periods=1).mean())
        agg_df['soil_mean_72h'] = grouped['soil_moisture_0_7cm'].transform(lambda x: x.rolling(72, min_periods=1).mean())
        agg_df['soil_mean_168h'] = grouped['soil_moisture_0_7cm'].transform(lambda x: x.rolling(168, min_periods=1).mean())
        
    for c in ['nino_34', 'mjo_phase']:
        if c in env_df.columns:
            agg_df[f'{c}_mean_720h'] = grouped[c].transform(lambda x: x.rolling(720, min_periods=1).mean())
            
    for c in ['temperature_c', 'wind_speed_kmh', 'surface_pressure_hpa', 'humidity_pct']:
        if c in env_df.columns:
            agg_df[f'{c}_mean_6h'] = grouped[c].transform(lambda x: x.rolling(6, min_periods=1).mean())

    # 📌 2. FITUR GLOBAL BASIN V25 (Pandangan Helikopter ke seluruh DAS)
    print("Menghitung Agregat DAS Bengawan Solo (Basin Features)...")
    basin_env = env_df.groupby('datetime').agg({
        'rainfall_mm': 'mean',
        'soil_moisture_0_7cm': 'mean'
    }).reset_index()

    basin_env['basin_rain_sum_72h'] = basin_env['rainfall_mm'].rolling(72, min_periods=1).sum()
    basin_env['basin_rain_sum_168h'] = basin_env['rainfall_mm'].rolling(168, min_periods=1).sum()
    basin_env['basin_rain_sum_360h'] = basin_env['rainfall_mm'].rolling(360, min_periods=1).sum()
    basin_env['basin_soil_mean_168h'] = basin_env['soil_moisture_0_7cm'].rolling(168, min_periods=1).mean()
    
    basin_features = basin_env[['datetime', 'basin_rain_sum_72h', 'basin_rain_sum_168h', 'basin_rain_sum_360h', 'basin_soil_mean_168h']]
    agg_df = pd.merge(agg_df, basin_features, on='datetime', how='left')
    
    # 📌 3. DATA PENDUKUNG (HydroRIVERS Static Features)
    hydro_mapping = {
        "Ketonggo": {"UPLAND_SKM": 21.5, "DIS_AV_CMS": 0.658, "ORD_STRA": 1},
        "Sekayu": {"UPLAND_SKM": 800.9, "DIS_AV_CMS": 30.149, "ORD_STRA": 4},
        "Cepu": {"UPLAND_SKM": 10926.8, "DIS_AV_CMS": 352.11, "ORD_STRA": 6},
        "Karanggeneng": {"UPLAND_SKM": 15087.4, "DIS_AV_CMS": 442.477, "ORD_STRA": 6},
        "Bojonegoro - Kali Kethek": {"UPLAND_SKM": 12778.1, "DIS_AV_CMS": 387.983, "ORD_STRA": 6},
        "Babat": {"UPLAND_SKM": 14342.2, "DIS_AV_CMS": 428.018, "ORD_STRA": 6},
        "Jurug": {"UPLAND_SKM": 11.3, "DIS_AV_CMS": 0.387, "ORD_STRA": 1},
        # Sisanya jika tidak ada di mapping akan default ke rata-rata atau 0
    }
    agg_df['upland_skm'] = agg_df['nama_pos'].apply(lambda x: hydro_mapping.get(x, {}).get('UPLAND_SKM', 0))
    agg_df['dis_av_cms'] = agg_df['nama_pos'].apply(lambda x: hydro_mapping.get(x, {}).get('DIS_AV_CMS', 0))
    agg_df['ord_stra'] = agg_df['nama_pos'].apply(lambda x: hydro_mapping.get(x, {}).get('ORD_STRA', 1))

    return agg_df

env_feat = build_exogenous_features(lingkungan_df)

def prepare_dataset(base_df, env_feat, historical_tma=None):
    df = base_df.copy()
    
    # 📌 Siklus Waktu Trigonometri
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
    
    # 📌 Injeksi Target Lag 365 Hari (Memory Musiman)
    if historical_tma is not None:
        historical_map = historical_tma.set_index(['nama_pos', 'datetime'])['tma_mdpl'].to_dict()
        df['tma_lag_365d'] = df.apply(lambda row: historical_map.get((row['nama_pos'], row['datetime'] - pd.DateOffset(years=1)), np.nan), axis=1)
        
        # Backward/Forward fill memori yang kosong pakai TMA terdekat terakhir
        last_known = historical_tma.dropna(subset=['tma_mdpl']).sort_values('datetime').groupby('nama_pos').tail(1).set_index('nama_pos')['tma_mdpl'].to_dict()
        df['tma_lag_365d'] = df['tma_lag_365d'].fillna(df['nama_pos'].map(last_known))
    
    return df

print("Menyusun Train & Test Set...")
historical_tma = train_df[['datetime', 'nama_pos', 'tma_mdpl']].copy()

train_full = prepare_dataset(train_df, env_feat, historical_tma)

test_parsed = test_df.copy()
split_info = test_parsed['id'].str.split(' - ', n=1, expand=True)
test_parsed['datetime'] = pd.to_datetime(split_info[0])
test_parsed['nama_pos'] = split_info[1]
test_full = prepare_dataset(test_parsed, env_feat, historical_tma)

train_full = train_full.dropna(subset=['tma_mdpl']).reset_index(drop=True)

drop_cols = ['datetime', 'nama_pos', 'tma_mdpl', 'id']
features = [c for c in train_full.columns if c not in drop_cols]
print(f"Total Fitur Final V37: {len(features)}")
print(f"Daftar Fitur: {features}")''')
    cells.append(cell2)

    # Cell 3: Local Training
    cell3 = nbf.v4.new_code_cell('''# ==============================================================================
# V37 OPTUNA + TIME-SERIES SPLIT + TWEEDIE (LOCAL GOAT)
# ==============================================================================
stations = train_full['nama_pos'].unique()
station_models = {}
station_scores = {}

optuna.logging.set_verbosity(optuna.logging.ERROR)
N_TRIALS = 15
N_SPLITS = 5

print(f"Memulai Optuna Local Training untuk 30 Stasiun (Total {30 * N_TRIALS} Trials)...\\n")

for st in stations:
    X_st = train_full[train_full['nama_pos'] == st][features]
    y_st = train_full[train_full['nama_pos'] == st]['tma_mdpl']
    
    def objective(trial):
        params = {
            'objective': 'reg:tweedie', 
            'tweedie_variance_power': trial.suggest_float('tweedie_variance_power', 1.1, 1.9),
            'eval_metric': 'rmse', 
            'tree_method': 'hist',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'n_estimators': 800,
            'random_state': 42
        }
        
        # Gunakan pembagian berurutan waktu (TimeSeries) untuk Validasi Optuna
        split_idx = int(len(X_st) * 0.8)
        X_tr, y_tr = X_st.iloc[:split_idx], np.clip(y_st.iloc[:split_idx], 0.0001, None)
        X_va, y_va = X_st.iloc[split_idx:], np.clip(y_st.iloc[split_idx:], 0.0001, None)
        
        model = xgb.XGBRegressor(**params, early_stopping_rounds=30)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        
        preds = model.predict(X_va)
        return np.sqrt(mean_squared_error(y_va, preds))

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS)
    
    best_params = study.best_params
    best_params['objective'] = 'reg:tweedie'
    best_params['eval_metric'] = 'rmse'
    best_params['tree_method'] = 'hist'
    best_params['n_estimators'] = 800
    best_params['random_state'] = 42

    # 📌 V37 Upgrade: TimeSeriesSplit untuk Validasi Anti-Bocor
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    models = []
    scores = []
    
    for train_idx, val_idx in tscv.split(X_st):
        X_tr, y_tr = X_st.iloc[train_idx], np.clip(y_st.iloc[train_idx], 0.0001, None)
        X_va, y_va = X_st.iloc[val_idx], np.clip(y_st.iloc[val_idx], 0.0001, None)
        
        model = xgb.XGBRegressor(**best_params, early_stopping_rounds=50)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        
        val_preds = model.predict(X_va)
        scores.append(np.sqrt(mean_squared_error(y_va, val_preds)))
        models.append(model)
        
    avg_rmse = np.mean(scores)
    station_models[st] = models
    station_scores[st] = avg_rmse
    
    print(f"[{st}] - Tweedie Power: {best_params['tweedie_variance_power']:.2f} | TimeSeries Val RMSE: {avg_rmse:.4f}")

global_rmse = np.mean(list(station_scores.values()))
print(f"\\n🏆 GRAND AVERAGE TIME-SERIES RMSE ACROSS 30 STATIONS: {global_rmse:.4f}")''')
    cells.append(cell3)

    # Cell 4: Evaluation
    cell4 = nbf.v4.new_code_cell('''# ==============================================================================
# EVALUASI & VISUALISASI
# ==============================================================================
TARGET_STATION = "Babat" # Ubah nama stasiun untuk melihat visualisasi lainnya

fig, ax = plt.subplots(1, 2, figsize=(22, 8))

feat_imp = np.mean([m.feature_importances_ for m in station_models[TARGET_STATION]], axis=0)
imp_df = pd.DataFrame({'Features': features, 'Importance score': feat_imp})
# Filter out features with 0 importance (like static data_pendukung in local models)
imp_df = imp_df[imp_df['Importance score'] > 0] 
imp_df = imp_df.sort_values(by='Importance score', ascending=False).head(20)

sns.barplot(x='Importance score', y='Features', data=imp_df, ax=ax[0], palette='viridis')
ax[0].set_title(f'Top Feature Importance [{TARGET_STATION}]')

X_target = train_full[train_full['nama_pos'] == TARGET_STATION][features]
y_target = train_full[train_full['nama_pos'] == TARGET_STATION]['tma_mdpl']

val_preds = np.zeros(len(X_target))
for m in station_models[TARGET_STATION]:
    val_preds += m.predict(X_target) / N_SPLITS

sub_actual = y_target.iloc[-250:].values
sub_pred = val_preds[-250:]
ax[1].plot(sub_actual, label='Actual TMA', color='blue', alpha=0.7, linewidth=2.5)
ax[1].plot(sub_pred, label='Predicted (V37 V25 Remastered)', color='red', linestyle='--', alpha=0.8, linewidth=2)
ax[1].set_title(f'Actual vs Predicted - 250 Hari Terakhir [{TARGET_STATION}]')
ax[1].legend()

plt.tight_layout()
plt.show()''')
    cells.append(cell4)

    # Cell 5: Submission
    cell5 = nbf.v4.new_code_cell('''# ==============================================================================
# KAGGLE SUBMISSION GENERATOR
# ==============================================================================
test_preds = np.zeros(len(test_full))

for st in stations:
    idx = test_full['nama_pos'] == st
    if idx.sum() == 0: continue
        
    X_test_st = test_full.loc[idx, features]
    
    st_preds = np.zeros(len(X_test_st))
    for m in station_models[st]:
        st_preds += m.predict(X_test_st) / N_SPLITS
        
    test_full.loc[idx, 'tma_mdpl'] = st_preds

sub_pred = test_full[['id', 'tma_mdpl']].copy()
sub = pd.merge(sample_sub[['id']], sub_pred, on='id', how='left')

# Mencegah nilai kosong di test set dengan pengisian batas aman
sub['tma_mdpl'] = sub['tma_mdpl'].fillna(method='ffill').fillna(method='bfill')

display(sub.head())
output_file = path + 'submission_v37_v25_remastered.csv'
sub.to_csv(output_file, index=False)
print(f"\\n✅ File '{output_file}' berhasil dicetak! Siap disubmit ke Kaggle Leaderboard!")''')
    cells.append(cell5)

    nb['cells'] = cells
    
    with open('V37_V25_Remastered.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    
    print("V37_V25_Remastered.ipynb berhasil dibuat!")

if __name__ == '__main__':
    create_v37_notebook()
