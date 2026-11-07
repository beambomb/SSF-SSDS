import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

def create_v20_notebook():
    nb = new_notebook()
    
    cells = [
        new_markdown_cell('# 🌲 V20 RANDOM FOREST ADVANCED FE 🌲\nMembawa seluruh kekuatan *Advanced Feature Engineering* dari V19 (VPD, Hidrologi, Spasial, Deteksi Badai) namun mengganti mesin utama dari algoritma *Boosting* (XGBoost) ke *Bagging* (**Random Forest**). Karena Random Forest membagi keputusan ke ribuan pohon secara buta dan mengambil rata-ratanya, algoritma ini secara alami sangat kebal terhadap *noise* cuaca liar dan dipastikan menghasilkan kurva prediksi yang sangat stabil!'),
        
        new_markdown_cell('## 1. Persiapan Environment & Pemuatan Data'),
        new_code_cell('''!pip install -q scikit-learn optuna scipy
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from xgboost import XGBRFRegressor
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

print("Data berhasil dimuat!")
'''),

        new_markdown_cell('## 2. Advanced Feature Engineering 🧠 (Kloning V19)\nKita menyuntikkan Termodinamika Penguapan (VPD), kecerdasan spasial tetangga, dan dinamika saturasi tanah.'),
        new_code_cell('''print("Mengekstraksi Fitur Fisika, Hidrologi, dan Spasial...")
lingkungan_df = lingkungan_df.sort_values(['nama_pos', 'datetime']).reset_index(drop=True)

# --- 2A. SPATIAL RAIN NETWORK ---
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
spatial_env = pd.concat(spatial_rain_list, ignore_index=True)

lingkungan_df = pd.merge(lingkungan_df, spatial_env, on=['datetime', 'nama_pos'], how='left')

# --- 2B. FISIKA & HIDROLOGI ---
def build_advanced_features(env_df):
    agg_df = env_df[['datetime', 'nama_pos', 'spatial_rain_avg']].copy()
    grouped = env_df.groupby('nama_pos')
    
    # Hujan & Iklim Global
    if 'rainfall_mm' in env_df.columns:
        agg_df['rain_sum_6h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(6, min_periods=1).sum())
        agg_df['rain_sum_72h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(72, min_periods=1).sum())
        agg_df['rain_sum_168h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(168, min_periods=1).sum())
        agg_df['rain_sum_360h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(360, min_periods=1).sum())
        agg_df['rain_sum_720h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(720, min_periods=1).sum())
    
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

    # 1. API & Effective Runoff
    if 'rainfall_mm' in env_df.columns:
        agg_df['api_14d'] = grouped['rainfall_mm'].transform(lambda x: x.ewm(alpha=0.07, adjust=False).mean())
        agg_df['effective_runoff'] = env_df['rainfall_mm'] * (env_df['soil_moisture_0_7cm'] / 100.0)
    
    # 2. Dinamika Badai
    if 'surface_pressure_hpa' in env_df.columns:
        agg_df['pressure_drop_24h'] = grouped['surface_pressure_hpa'].transform(lambda x: x.diff(8).fillna(0))
    if 'rainfall_mm' in env_df.columns:
        agg_df['rain_max_24h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(8, min_periods=1).max())
    if 'soil_moisture_0_7cm' in env_df.columns:
        agg_df['soil_trend'] = agg_df['soil_mean_6h'] - agg_df['soil_mean_72h']

    # 3. Termodinamika Penguapan (VPD)
    if 'temperature_c' in env_df.columns and 'humidity_pct' in env_df.columns:
        T = env_df['temperature_c']
        RH = env_df['humidity_pct']
        es = 0.6108 * np.exp(17.27 * T / (T + 237.3))
        ea = es * (RH / 100.0)
        agg_df['vpd_kpa'] = es - ea

    return agg_df

env_feat = build_advanced_features(lingkungan_df)

station_elevation = train_df.groupby('nama_pos')['tma_mdpl'].mean().reset_index()
station_elevation.rename(columns={'tma_mdpl': 'elevation_proxy'}, inplace=True)

def prepare_dataset(base_df, env_feat, coords_df, elevation_df):
    df = base_df.copy()
    
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['dayofyear'] = df['datetime'].dt.dayofyear
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12.0)
    
    df = pd.merge(df, env_feat, on=['datetime', 'nama_pos'], how='left')
    df = pd.merge(df, coords_df, on='nama_pos', how='left')
    df = pd.merge(df, elevation_df, on='nama_pos', how='left')
    
    return df

print("Menyusun Dataset Gabungan...")
train_full = prepare_dataset(train_df, env_feat, koordinat_df, station_elevation)

test_parsed = test_df.copy()
split_info = test_parsed['id'].str.split(' - ', n=1, expand=True)
test_parsed['datetime'] = pd.to_datetime(split_info[0])
test_parsed['nama_pos'] = split_info[1]
test_full = prepare_dataset(test_parsed, env_feat, koordinat_df, station_elevation)

le = LabelEncoder()
all_stations = pd.concat([train_full['nama_pos'], test_full['nama_pos']]).unique()
le.fit(all_stations)

train_full['nama_pos_encoded'] = le.transform(train_full['nama_pos'])
test_full['nama_pos_encoded'] = le.transform(test_full['nama_pos'])

train_full = train_full.dropna(subset=['tma_mdpl']).reset_index(drop=True)

drop_cols = ['datetime', 'nama_pos', 'tma_mdpl', 'id']
features = [c for c in train_full.columns if c not in drop_cols]
print(f"✅ Total Fitur: {len(features)} Fitur Siap Dimasukkan ke Hutan Demokrasi!")

# Hapus nilai NaN yang mungkin masih tersisa dari rolling/diff di fitur-fitur awal
train_full = train_full.dropna(subset=features).reset_index(drop=True)
'''),

        new_markdown_cell('## 3. Optuna Hyperparameter Tuning Random Forest 🤖\nMencari parameter terbaik untuk Random Forest. Karena RF cukup berat, kita *set* *trials* tidak terlalu banyak, namun cukup untuk menemukan titik kestabilan. (Perlu diketahui RF tidak butuh *early stopping*).'),
        new_code_cell('''X = train_full[features]
y = train_full['tma_mdpl']

# Isi NaN dengan nilai rata-rata kolom jika ada yang bocor (RF scikit-learn tidak bisa handle NaN)
X = X.fillna(X.mean())

RUN_OPTUNA = True
N_TRIALS = 15 

if RUN_OPTUNA:
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'colsample_bynode': trial.suggest_float('colsample_bynode', 0.5, 1.0),
            'learning_rate': 1.0, # RF di XGBoost selalu pakai lr=1
            'tree_method': 'hist',
            'random_state': 42
        }
        
        # Hold-out validation (20%) untuk Optuna
        split_idx = int(len(X) * 0.8)
        X_tr, y_tr = X.iloc[:split_idx], y.iloc[:split_idx]
        X_va, y_va = X.iloc[split_idx:], y.iloc[split_idx:]
        
        model = XGBRFRegressor(**params)
        model.fit(X_tr, y_tr)
        
        preds = model.predict(X_va)
        rmse = root_mean_squared_error(y_va, preds)
        return rmse

    print(f"Memulai Optuna Random Forest ({N_TRIALS} Trials)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS)
    
    print("\\nBest parameters found by Optuna:")
    best_params = study.best_params
    print(best_params)
else:
    best_params = {
        'n_estimators': 150, 'max_depth': 12, 'colsample_bynode': 0.8,
        'learning_rate': 1.0, 'tree_method': 'hist', 'random_state': 42
    }

'''),

        new_markdown_cell('## 4. K-Fold Training (5-Fold Shuffle=True)\nMelatih Random Forest dengan 5-Fold *Cross Validation*. Hutan keputusan ini akan menangkap pola global tanpa *overfitting*!'),
        new_code_cell('''kf = KFold(n_splits=5, shuffle=True, random_state=42)
models = []
scores = []

print("Memulai K-Fold Training Random Forest...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"--- Fold {fold+1} ---")
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    model = XGBRFRegressor(**best_params)
    model.fit(X_tr, y_tr)
    
    val_preds = model.predict(X_va)
    fold_rmse = root_mean_squared_error(y_va, val_preds)
    scores.append(fold_rmse)
    models.append(model)
    print(f"✅ Fold {fold+1} RMSE: {fold_rmse:.4f}\\n")

print(f"🏆 Rata-rata CV RMSE (K-Fold): {np.mean(scores):.4f}")
'''),

        new_markdown_cell('## 5. Visualisasi Karakter Random Forest 🌲\nPerhatikan bagaimana fitur *Advanced Engineering* dihargai oleh Random Forest, dan lihat betapa stabilnya tebakan RF (bebas dari *noise* liar).'),
        new_code_cell('''fig, ax = plt.subplots(1, 2, figsize=(22, 8))

# 1. Feature Importance (Random Forest)
feat_imp = np.mean([m.feature_importances_ for m in models], axis=0)
imp_df = pd.DataFrame({'Features': features, 'Importance score': feat_imp})
imp_df = imp_df.sort_values(by='Importance score', ascending=False).head(20)

sns.barplot(x='Importance score', y='Features', data=imp_df, ax=ax[0], palette='crest')
ax[0].set_title('Top 20 Feature Importance (Random Forest)')

# 2. Actual vs Predicted (Validasi Fold Terakhir)
sub_actual = y_va.iloc[-200:].values
sub_pred = val_preds[-200:]
ax[1].plot(sub_actual, label='Actual TMA', color='blue', alpha=0.7, linewidth=2)
ax[1].plot(sub_pred, label='RF Predicted', color='green', alpha=0.7, linewidth=2)
ax[1].set_title('Actual vs Predicted (Validation Set - Sampel 200)')
ax[1].legend()

plt.tight_layout()
plt.show()
'''),

        new_markdown_cell('## 6. Inferensi Hutan (Zero Data Leakage) ⚡\nSetiap prediksi test set di-*voting* oleh 5 model *Random Forest* (yang masing-masing memuat ratusan pohon). Stabilitas absolut!'),
        new_code_cell('''X_test = test_full[features]
X_test = X_test.fillna(X.mean()) # Imputasi jika ada NaN

test_preds = np.zeros(len(X_test))
for model in models:
    test_preds += model.predict(X_test) / len(models)

test_full['tma_mdpl'] = test_preds

sub_pred = test_full[['id', 'tma_mdpl']].copy()
sub = pd.merge(sample_sub[['id']], sub_pred, on='id', how='left')

sub['tma_mdpl'] = sub['tma_mdpl'].fillna(method='ffill').fillna(method='bfill')

print("Sample Submission:")
display(sub.head())

sub.to_csv('submission_v20_randomforest_advanced_fe.csv', index=False)
print("\\n✅ File 'submission_v20_randomforest_advanced_fe.csv' berhasil dibuat! Silakan uji stabilitas RF di Leaderboard!")
''')
    ]
    
    nb.cells.extend(cells)
    with open('V20_RandomForest_Advanced_FE.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == '__main__':
    create_v20_notebook()
    print("Notebook V20_RandomForest_Advanced_FE.ipynb berhasil dibuat!")
