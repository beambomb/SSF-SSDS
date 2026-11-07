import nbformat as nbf

def create_v36_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # Cell 1: Administrasi & Persiapan Data
    cell1 = nbf.v4.new_code_cell('''# ==============================================================================
# V36: NORMALIZED GLOBAL ENSEMBLE (THE KAGGLE GOLD STANDARD)
# Fitur: TimeSeriesSplit, Z-Score Target Normalization, Stacking Meta-Learner, 
#        Siklus Trigonometri, & EWM Hydrology.
# ==============================================================================

import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')

# 1. SETUP LINGKUNGAN (GOOGLE COLAB / LOKAL)
try:
    from google.colab import drive
    drive.mount('/content/drive')
    base_path = '/content/drive/MyDrive/SSF-SSDS'
    print("Berjalan di Google Colab. Drive berhasil di-mount.")
except ImportError:
    base_path = '.'
    print("Berjalan di environment lokal.")

def get_path(filename):
    return os.path.join(base_path, filename)

# 2. MEMUAT SEMUA DATASET
print("\\nMemuat dataset...")
train_df = pd.read_csv(get_path('train.csv'))
test_df = pd.read_csv(get_path('test.csv'))
lingkungan_df = pd.read_csv(get_path('data_lingkungan.csv'))
koordinat_df = pd.read_csv(get_path('data_pendukung/koordinat_pos.csv'))

train_df['datetime'] = pd.to_datetime(train_df['datetime'])
lingkungan_df['datetime'] = pd.to_datetime(lingkungan_df['datetime'])

print(f"Data Train: {train_df.shape}")
print(f"Data Test : {test_df.shape}")
print(f"Data Lingkungan: {lingkungan_df.shape}")''')
    cells.append(cell1)

    # Cell 2: Feature Engineering & Preprocessing
    cell2 = nbf.v4.new_code_cell('''# ==============================================================================
# FEATURE ENGINEERING & Z-SCORE NORMALIZATION
# ==============================================================================
print("Mengekstraksi Fitur Fisika Hidrologi & Siklus Waktu...")

# 1. Haversine Matrix untuk Hujan Kiriman Spasial
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    a = np.sin(dLat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dLon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

coord_dict = koordinat_df.set_index('nama_pos').to_dict('index')
dist_matrix = {p1: {p2: 0 if p1 == p2 else haversine(coord_dict[p1]['latitude'], coord_dict[p1]['longitude'], coord_dict[p2]['latitude'], coord_dict[p2]['longitude']) for p2 in coord_dict} for p1 in coord_dict}

# 2. EWM & Siklus (The 40 Features)
lingkungan_df = lingkungan_df.sort_values(['nama_pos', 'datetime']).reset_index(drop=True)
env_feat = lingkungan_df.copy()
grouped = env_feat.groupby('nama_pos')

# Hujan terserap (API EWM)
if 'rainfall_mm' in env_feat.columns:
    env_feat['rain_ewm_48h'] = grouped['rainfall_mm'].transform(lambda x: x.ewm(halflife=48, ignore_na=True).mean())
    env_feat['rain_ewm_168h'] = grouped['rainfall_mm'].transform(lambda x: x.ewm(halflife=168, ignore_na=True).mean())
    env_feat['rain_sum_24h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(24, min_periods=1).sum())
    
    # Simple soil moisture aggregation to calculate effective runoff
    soil_cols = ['soil_moisture_0_7cm', 'soil_moisture_7_28cm', 'soil_moisture_28_100cm', 'soil_moisture_100_255cm']
    if all(c in env_feat.columns for c in soil_cols):
        total_soil = env_feat[soil_cols].mean(axis=1)
        sat_ratio = total_soil / total_soil.max()
        env_feat['effective_runoff'] = env_feat['rainfall_mm'] * sat_ratio
        env_feat['eff_runoff_sum_72h'] = grouped['effective_runoff'].transform(lambda x: x.rolling(72, min_periods=1).sum())

# Hujan Spasial (Kiriman Hulu)
if 'rain_sum_24h' in env_feat.columns:
    rain_pivot = env_feat.pivot(index='datetime', columns='nama_pos', values='rain_sum_24h').fillna(0)
    spatial_features = []
    for pos in rain_pivot.columns:
        weights = pd.Series({p2: np.exp(-dist_matrix[pos][p2]/75.0) for p2 in rain_pivot.columns})
        weights[pos] = 0
        weighted_rain = (rain_pivot * weights).sum(axis=1)
        spatial_features.append(pd.DataFrame({'datetime': rain_pivot.index, 'nama_pos': pos, 'spatial_rain_sum_24h': weighted_rain.values}))
    env_feat = pd.merge(env_feat, pd.concat(spatial_features, ignore_index=True), on=['datetime', 'nama_pos'], how='left')

# Siklus Trigonometri
env_feat['month'] = env_feat['datetime'].dt.month
env_feat['dayofyear'] = env_feat['datetime'].dt.dayofyear
env_feat['hour'] = env_feat['datetime'].dt.hour

env_feat['month_sin'] = np.sin(2 * np.pi * env_feat['month']/12.0)
env_feat['month_cos'] = np.cos(2 * np.pi * env_feat['month']/12.0)
env_feat['dayofyear_sin'] = np.sin(2 * np.pi * env_feat['dayofyear']/365.25)
env_feat['dayofyear_cos'] = np.cos(2 * np.pi * env_feat['dayofyear']/365.25)
env_feat['hour_sin'] = np.sin(2 * np.pi * env_feat['hour']/24.0)
env_feat['hour_cos'] = np.cos(2 * np.pi * env_feat['hour']/24.0)
env_feat.drop(columns=['month', 'dayofyear', 'hour'], inplace=True)

# HydroRIVERS (Fitur Kapasitas Topologi)
hydro_mapping = {
    "Ketonggo": {"UPLAND_SKM": 21.5, "DIS_AV_CMS": 0.658, "ORD_STRA": 1},
    "Sekayu": {"UPLAND_SKM": 800.9, "DIS_AV_CMS": 30.149, "ORD_STRA": 4},
    "Cepu": {"UPLAND_SKM": 10926.8, "DIS_AV_CMS": 352.11, "ORD_STRA": 6},
    "Karanggeneng": {"UPLAND_SKM": 15087.4, "DIS_AV_CMS": 442.477, "ORD_STRA": 6},
    "Bojonegoro - Kali Kethek": {"UPLAND_SKM": 12778.1, "DIS_AV_CMS": 387.983, "ORD_STRA": 6},
    "Babat": {"UPLAND_SKM": 14342.2, "DIS_AV_CMS": 428.018, "ORD_STRA": 6}
}
env_feat['upland_skm'] = env_feat['nama_pos'].apply(lambda x: hydro_mapping.get(x, {}).get('UPLAND_SKM', 0))
env_feat['dis_av_cms'] = env_feat['nama_pos'].apply(lambda x: hydro_mapping.get(x, {}).get('DIS_AV_CMS', 0))
env_feat['ord_stra'] = env_feat['nama_pos'].apply(lambda x: hydro_mapping.get(x, {}).get('ORD_STRA', 1))

# 3. Penggabungan Data
train_full = pd.merge(train_df, env_feat, on=['datetime', 'nama_pos'], how='left')
test_parsed = test_df.copy()
split_info = test_parsed['id'].str.split(' - ', n=1, expand=True)
test_parsed['datetime'] = pd.to_datetime(split_info[0])
test_parsed['nama_pos'] = split_info[1]
test_full = pd.merge(test_parsed, env_feat, on=['datetime', 'nama_pos'], how='left')
test_full['tma_mdpl'] = np.nan

# 4. Target Lag 365 Hari (Memory Tahunan)
historical_map = train_df.set_index(['nama_pos', 'datetime'])['tma_mdpl'].to_dict()
train_full['tma_lag_365d'] = train_full.apply(lambda row: historical_map.get((row['nama_pos'], row['datetime'] - pd.DateOffset(years=1)), np.nan), axis=1)
test_full['tma_lag_365d'] = test_full.apply(lambda row: historical_map.get((row['nama_pos'], row['datetime'] - pd.DateOffset(years=1)), np.nan), axis=1)
last_known = train_df.dropna(subset=['tma_mdpl']).sort_values('datetime').groupby('nama_pos').tail(1).set_index('nama_pos')['tma_mdpl'].to_dict()
train_full['tma_lag_365d'] = train_full['tma_lag_365d'].fillna(train_full['nama_pos'].map(last_known))
test_full['tma_lag_365d'] = test_full['tma_lag_365d'].fillna(test_full['nama_pos'].map(last_known))

# 5. Z-SCORE ROBUST TARGET NORMALIZATION (RAHASIA UTAMA 0.81)
print("\\nMenghitung Z-Score Target (Median & IQR per stasiun)...")
target_stats = {}
train_full['tma_zscore'] = np.nan

for pos in train_full['nama_pos'].unique():
    pos_data = train_full[train_full['nama_pos'] == pos]['tma_mdpl'].dropna()
    median_val = pos_data.median()
    q1 = pos_data.quantile(0.25)
    q3 = pos_data.quantile(0.75)
    iqr_val = q3 - q1
    scale = max(iqr_val / 1.349, 0.05) # Mencegah pembagian 0
    min_val = pos_data.min() # Untuk clipping nanti
    
    target_stats[pos] = {'median': median_val, 'scale': scale, 'min': min_val}
    
    idx = train_full['nama_pos'] == pos
    train_full.loc[idx, 'tma_zscore'] = (train_full.loc[idx, 'tma_mdpl'] - median_val) / scale

# 6. Pemilihan Fitur
drop_cols = ['datetime', 'id', 'nama_pos', 'tma_mdpl', 'tma_zscore']
# Target Encoding untuk nama_pos (hanya angka yg masuk fit)
mean_enc = train_full.groupby('nama_pos')['tma_zscore'].mean()
train_full['pos_encoded'] = train_full['nama_pos'].map(mean_enc)
test_full['pos_encoded'] = test_full['nama_pos'].map(mean_enc)

numeric_cols = train_full.select_dtypes(include=[np.number]).columns
features = [c for c in numeric_cols if c not in drop_cols]
print(f"Total Fitur Super: {len(features)}")
print(features)''')
    cells.append(cell2)

    # Cell 3: TimeSeriesSplit & Stacked Generalization Training
    cell3 = nbf.v4.new_code_cell('''# ==============================================================================
# TIME-SERIES SPLIT & STACKED GENERALIZATION TRAINING
# ==============================================================================
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

print("Memulai Proses Training Ensemble Global (27.000 Baris)...")

X = train_full[features].fillna(0) # XGB & LGBM handle 0/NaN well
y = train_full['tma_zscore']

# Siapkan array untuk Out-Of-Fold (OOF) predictions (untuk melatih Meta-Learner)
oof_xgb = np.zeros(len(X))
oof_lgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))

test_X = test_full[features].fillna(0)
preds_xgb = np.zeros(len(test_X))
preds_lgb = np.zeros(len(test_X))
preds_cat = np.zeros(len(test_X))

models_xgb, models_lgb, models_cat = [], [], []

# TimeSeriesSplit (5 Folds) untuk mencegah Data Leakage masa depan
tscv = TimeSeriesSplit(n_splits=5)

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    print(f"\\n--- Fold {fold+1}/5 ---")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # 1. XGBoost Regressor (RMSE loss)
    model_xgb = xgb.XGBRegressor(
        n_estimators=1000, learning_rate=0.03, max_depth=6, 
        reg_lambda=2.0, subsample=0.8, colsample_bytree=0.8,
        tree_method='hist', random_state=42 + fold,
        early_stopping_rounds=50, n_jobs=-1
    )
    model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx] = model_xgb.predict(X_val)
    preds_xgb += model_xgb.predict(test_X) / 5
    models_xgb.append(model_xgb)
    rmse_xgb = np.sqrt(mean_squared_error(y_val, oof_xgb[val_idx]))
    print(f"  [XGB] Val Z-Score RMSE: {rmse_xgb:.4f}")
    
    # 2. LightGBM Regressor
    model_lgb = lgb.LGBMRegressor(
        n_estimators=1000, learning_rate=0.03, max_depth=6,
        reg_lambda=2.0, subsample=0.8, colsample_bytree=0.8,
        random_state=123 + fold, n_jobs=-1
    )
    model_lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_lgb[val_idx] = model_lgb.predict(X_val)
    preds_lgb += model_lgb.predict(test_X) / 5
    models_lgb.append(model_lgb)
    rmse_lgb = np.sqrt(mean_squared_error(y_val, oof_lgb[val_idx]))
    print(f"  [LGB] Val Z-Score RMSE: {rmse_lgb:.4f}")
    
    # 3. CatBoost Regressor
    model_cat = CatBoostRegressor(
        iterations=1000, learning_rate=0.03, depth=6,
        l2_leaf_reg=3.0, random_seed=777 + fold, verbose=0
    )
    model_cat.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
    oof_cat[val_idx] = model_cat.predict(X_val)
    preds_cat += model_cat.predict(test_X) / 5
    models_cat.append(model_cat)
    rmse_cat = np.sqrt(mean_squared_error(y_val, oof_cat[val_idx]))
    print(f"  [CAT] Val Z-Score RMSE: {rmse_cat:.4f}")

print("\\n[META-LEARNER] Melatih Ridge Regression untuk Stacking Ensemble...")
# Kita latih Meta-Learner hanya pada lipatan terakhir dari OOF (karena data paling awal banyak nol)
valid_idx = np.where(oof_xgb != 0)[0]
OOF_train = np.column_stack((oof_xgb[valid_idx], oof_lgb[valid_idx], oof_cat[valid_idx]))
Y_valid = y.iloc[valid_idx]

meta_model = RidgeCV(alphas=[0.1, 1.0, 10.0])
meta_model.fit(OOF_train, Y_valid)

print(f"Bobot Ensemble Optimal (XGB, LGBM, CAT): {meta_model.coef_}")
print(f"Intercept: {meta_model.intercept_:.4f}")

OOF_meta = meta_model.predict(OOF_train)
meta_rmse = np.sqrt(mean_squared_error(Y_valid, OOF_meta))
print(f"\\nFINAL Z-SCORE ENSEMBLE RMSE: {meta_rmse:.4f}")

# Memprediksi Test Set menggunakan Meta-Learner
TEST_meta_features = np.column_stack((preds_xgb, preds_lgb, preds_cat))
final_zscore_preds = meta_model.predict(TEST_meta_features)''')
    cells.append(cell3)

    # Cell 4: Inverse Transformation & Submission
    cell4 = nbf.v4.new_code_cell('''# ==============================================================================
# INVERSE TRANSFORMATION & CLIPPING (SARAN 6)
# ==============================================================================
print("Mengubah Z-Score kembali menjadi Meter (Inverse Transform)...")

test_full['tma_zscore_pred'] = final_zscore_preds
test_full['tma_mdpl_pred'] = 0.0

for pos in test_full['nama_pos'].unique():
    stats = target_stats[pos]
    idx = test_full['nama_pos'] == pos
    
    # Inverse Z-Score formula
    raw_pred = (test_full.loc[idx, 'tma_zscore_pred'] * stats['scale']) + stats['median']
    
    # Clipping batas fisik historis terendah
    clipped_pred = raw_pred.clip(lower=stats['min'])
    test_full.loc[idx, 'tma_mdpl_pred'] = clipped_pred

# Buat Submission
sub = pd.DataFrame({
    'id': test_df['id'],
    'tma_mdpl': test_full['tma_mdpl_pred'].values
})

sub.to_csv(get_path('submission_v36_normalized_global.csv'), index=False)
print(f"Berhasil! File disimpan di: {get_path('submission_v36_normalized_global.csv')}")
display(sub.head())''')
    cells.append(cell4)

    # Cell 5: Visualisasi Train (Actual vs Predicted & Residuals)
    cell5 = nbf.v4.new_code_cell('''# ==============================================================================
# VISUALISASI 1: ACTUAL VS PREDICTED (TRAIN SET) + RESIDUAL PLOT (SARAN 10)
# ==============================================================================
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

# Kita hitung prediksi train keseluruhan untuk divisualisasikan
print("Mengkalkulasi prediksi Train untuk visualisasi...")
X_train_all = train_full[features].fillna(0)
train_xgb_pred = np.mean([m.predict(X_train_all) for m in models_xgb], axis=0)
train_lgb_pred = np.mean([m.predict(X_train_all) for m in models_lgb], axis=0)
train_cat_pred = np.mean([m.predict(X_train_all) for m in models_cat], axis=0)

TRAIN_meta_features = np.column_stack((train_xgb_pred, train_lgb_pred, train_cat_pred))
train_zscore_pred = meta_model.predict(TRAIN_meta_features)

train_full['tma_mdpl_pred'] = 0.0
for pos in train_full['nama_pos'].unique():
    stats = target_stats[pos]
    idx = train_full['nama_pos'] == pos
    train_full.loc[idx, 'tma_mdpl_pred'] = (train_zscore_pred[idx] * stats['scale']) + stats['median']

def plot_train_eval(stasiun):
    df_st = train_full[train_full['nama_pos'] == stasiun].sort_values('datetime')
    df_st = df_st.set_index('datetime')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Actual vs Predicted
    ax1.plot(df_st.index, df_st['tma_mdpl'], label='Actual (Meter)', color='#2ecc71', linewidth=2)
    ax1.plot(df_st.index, df_st['tma_mdpl_pred'], label='Predicted V36 (Meter)', color='#e74c3c', linestyle='--', linewidth=2)
    ax1.set_title(f'Evaluasi Training Set (V36) - Stasiun: {stasiun}', fontsize=14)
    ax1.set_ylabel('TMA (MDPL)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Residual Plot (Saran 10)
    residuals = df_st['tma_mdpl'] - df_st['tma_mdpl_pred']
    ax2.scatter(df_st.index, residuals, color='#3498db', alpha=0.6, s=15)
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.set_title('Residual Plot (Actual - Predicted)', fontsize=10)
    ax2.set_ylabel('Error (Meter)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

dropdown_stasiun_train = widgets.Dropdown(options=sorted(train_full['nama_pos'].unique()), description='Stasiun:', value='Babat')
widgets.interactive(plot_train_eval, stasiun=dropdown_stasiun_train)''')
    cells.append(cell5)
    
    # Cell 6: Visualisasi Forecasting Test Set
    cell6 = nbf.v4.new_code_cell('''# ==============================================================================
# VISUALISASI 2: 8-MONTH FORECASTING PROJECTION (TEST SET)
# ==============================================================================
def plot_forecast_test(stasiun):
    df_st = test_full[test_full['nama_pos'] == stasiun].sort_values('datetime')
    df_st = df_st.set_index('datetime')
    
    # Cari nilai maksimum historis untuk perbandingan
    hist_max = train_full[train_full['nama_pos'] == stasiun]['tma_mdpl'].max()
    
    plt.figure(figsize=(15, 5))
    plt.plot(df_st.index, df_st['tma_mdpl_pred'], label='V36 Forecast (Meter)', color='#9b59b6', linewidth=2.5)
    plt.axhline(hist_max, color='#e74c3c', linestyle=':', linewidth=2, label=f'Max Historis ({hist_max:.2f}m)')
    
    plt.title(f'8-Month Future Forecast - Stasiun: {stasiun}', fontsize=14)
    plt.xlabel('Bulan (2025 - 2026)')
    plt.ylabel('TMA (MDPL)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

print("Melihat proyeksi tebakan model di Kaggle Test Set:")
dropdown_stasiun_test = widgets.Dropdown(options=sorted(test_full['nama_pos'].unique()), description='Stasiun:', value='Babat')
widgets.interactive(plot_forecast_test, stasiun=dropdown_stasiun_test)''')
    cells.append(cell6)

    nb['cells'] = cells
    
    with open('V36_Normalized_Global.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    
    print("V36_Normalized_Global.ipynb berhasil dibuat!")

if __name__ == '__main__':
    create_v36_notebook()