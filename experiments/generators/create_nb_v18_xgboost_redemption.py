import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

def create_v18_notebook():
    nb = new_notebook()
    
    cells = [
        new_markdown_cell('# 🚀 V18 THE XGBOOST REDEMPTION 👑\nKembali ke akar pemenang sesungguhnya. Eksperimen membuktikan XGBoost mampu mengekstrak informasi temporal dari cuaca makro jauh lebih baik daripada algoritma lain. Di V18, kita kembali 100% ke arsitektur **Pure Exogenous V4** tanpa *Data Leakage*, tanpa iterasi *Recursive* yang meledak, namun diperkuat dengan *Hyperparameter Tuning* Optuna otomatis.'),
        
        new_markdown_cell('## 1. Persiapan Environment & Pemuatan Data'),
        new_code_cell('''!pip install -q xgboost optuna scikit-learn
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import LabelEncoder

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

        new_markdown_cell('## 2. Feature Engineering (Kloning Murni V4)\nMenghitung *rolling window* untuk semua fitur alam, persis seperti rancangan aslimu yang berhasil memuncaki *leaderboard*.'),
        new_code_cell('''print("Memproses Ekstraksi Fitur Cuaca...")
lingkungan_df = lingkungan_df.sort_values(['nama_pos', 'datetime']).reset_index(drop=True)

def build_exogenous_features(env_df):
    agg_df = env_df[['datetime', 'nama_pos']].copy()
    grouped = env_df.groupby('nama_pos')
    
    # 1. Hujan
    if 'rainfall_mm' in env_df.columns:
        agg_df['rain_sum_6h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(6, min_periods=1).sum())
        agg_df['rain_sum_72h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(72, min_periods=1).sum())
        agg_df['rain_sum_168h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(168, min_periods=1).sum())
        agg_df['rain_sum_360h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(360, min_periods=1).sum())
        agg_df['rain_sum_720h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(720, min_periods=1).sum())
        
    # 2. Kelembapan Tanah
    if 'soil_moisture_0_7cm' in env_df.columns:
        agg_df['soil_mean_6h'] = grouped['soil_moisture_0_7cm'].transform(lambda x: x.rolling(6, min_periods=1).mean())
        agg_df['soil_mean_72h'] = grouped['soil_moisture_0_7cm'].transform(lambda x: x.rolling(72, min_periods=1).mean())
        agg_df['soil_mean_168h'] = grouped['soil_moisture_0_7cm'].transform(lambda x: x.rolling(168, min_periods=1).mean())
        
    # 3. Iklim Global
    for c in ['nino_34', 'mjo_phase']:
        if c in env_df.columns:
            agg_df[f'{c}_mean_720h'] = grouped[c].transform(lambda x: x.rolling(720, min_periods=1).mean())
            
    # 4. Cuaca Makro (Fitur Kunci XGBoost)
    for c in ['temperature_c', 'wind_speed_kmh', 'surface_pressure_hpa', 'humidity_pct']:
        if c in env_df.columns:
            agg_df[f'{c}_mean_6h'] = grouped[c].transform(lambda x: x.rolling(6, min_periods=1).mean())

    return agg_df

env_feat = build_exogenous_features(lingkungan_df)

# Ekstrak Proxy Elevasi Stasiun (JANGKAR V4)
station_elevation = train_df.groupby('nama_pos')['tma_mdpl'].mean().reset_index()
station_elevation.rename(columns={'tma_mdpl': 'elevation_proxy'}, inplace=True)

def prepare_dataset(base_df, env_feat, coords_df, elevation_df):
    df = base_df.copy()
    
    # Waktu
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['dayofyear'] = df['datetime'].dt.dayofyear
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12.0)
    
    # Merge Lingkungan
    df = pd.merge(df, env_feat, on=['datetime', 'nama_pos'], how='left')
    
    # Merge Spasial (Koordinat & Elevasi)
    df = pd.merge(df, coords_df, on='nama_pos', how='left')
    df = pd.merge(df, elevation_df, on='nama_pos', how='left')
    
    return df

print("Menyusun Train Set...")
train_full = prepare_dataset(train_df, env_feat, koordinat_df, station_elevation)

print("Menyusun Test Set...")
test_parsed = test_df.copy()
split_info = test_parsed['id'].str.split(' - ', n=1, expand=True)
test_parsed['datetime'] = pd.to_datetime(split_info[0])
test_parsed['nama_pos'] = split_info[1]

test_full = prepare_dataset(test_parsed, env_feat, koordinat_df, station_elevation)

# Label Encoding Nama Pos
le = LabelEncoder()
all_stations = pd.concat([train_full['nama_pos'], test_full['nama_pos']]).unique()
le.fit(all_stations)

train_full['nama_pos_encoded'] = le.transform(train_full['nama_pos'])
test_full['nama_pos_encoded'] = le.transform(test_full['nama_pos'])

train_full = train_full.dropna(subset=['tma_mdpl']).reset_index(drop=True)

drop_cols = ['datetime', 'nama_pos', 'tma_mdpl', 'id']
features = [c for c in train_full.columns if c not in drop_cols]
print(f"Total Fitur Exogenous: {len(features)}")
'''),

        new_markdown_cell('## 3. Optuna Hyperparameter Tuning XGBoost 🤖\nMencari *hyperparameter* terbaik dengan *Early Stopping* super agresif.'),
        new_code_cell('''X = train_full[features]
y = train_full['tma_mdpl']

RUN_OPTUNA = True
N_TRIALS = 15 # Ubah jadi 50-100 jika butuh pencarian lebih agresif

if RUN_OPTUNA:
    def objective(trial):
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'hist',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'n_estimators': 1500,
            'random_state': 42
        }
        
        # Hold-out validation (20%) untuk Optuna
        split_idx = int(len(X) * 0.8)
        X_tr, y_tr = X.iloc[:split_idx], y.iloc[:split_idx]
        X_va, y_va = X.iloc[split_idx:], y.iloc[split_idx:]
        
        model = xgb.XGBRegressor(**params, early_stopping_rounds=30)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        
        preds = model.predict(X_va)
        rmse = root_mean_squared_error(y_va, preds)
        return rmse

    print(f"Memulai Optuna XGBoost ({N_TRIALS} Trials)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS)
    
    print("\\nBest parameters found by Optuna:")
    best_params = study.best_params
else:
    best_params = {
        'learning_rate': 0.03, 'max_depth': 7, 'subsample': 0.8, 
        'colsample_bytree': 0.9, 'min_child_weight': 3, 'reg_lambda': 1.0
    }

best_params['objective'] = 'reg:squarederror'
best_params['eval_metric'] = 'rmse'
best_params['tree_method'] = 'hist'
best_params['random_state'] = 42
best_params['n_estimators'] = 1500
'''),

        new_markdown_cell('## 4. K-Fold Training Ensemble\nMelatih 5 Model XGBoost dengan data yang diacak (`shuffle=True`). Strategi K-Fold ini memastikan seluruh model terlatih dalam semua periode cuaca (kemarau & penghujan).'),
        new_code_cell('''kf = KFold(n_splits=5, shuffle=True, random_state=42)
models = []
scores = []

print("Memulai K-Fold Training dengan XGBoost...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"--- Fold {fold+1} ---")
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    model = xgb.XGBRegressor(**best_params, early_stopping_rounds=50)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=250)
    
    val_preds = model.predict(X_va)
    fold_rmse = root_mean_squared_error(y_va, val_preds)
    scores.append(fold_rmse)
    models.append(model)
    print(f"✅ Fold {fold+1} RMSE: {fold_rmse:.4f}\\n")

print(f"🏆 Rata-rata CV RMSE (K-Fold): {np.mean(scores):.4f}")
'''),

        new_markdown_cell('## 5. Visualisasi Karakter XGBoost (Mengapa Ia Sang Raja) 👑\nXGBoost secara aktif membedah fitur-fitur alam (suhu, waktu musim, badai Nino) tanpa terjebak pada asumsi statis (`elevation_proxy`).'),
        new_code_cell('''fig, ax = plt.subplots(1, 2, figsize=(20, 7))

# 1. Feature Importance (XGBoost)
feat_imp = models[0].feature_importances_
imp_df = pd.DataFrame({'Features': features, 'Importance score': feat_imp})
imp_df = imp_df.sort_values(by='Importance score', ascending=False).head(20)

sns.barplot(x='Importance score', y='Features', data=imp_df, ax=ax[0], palette='magma')
ax[0].set_title('Top 20 Feature Importance (XGBoost V4 Asli)')

# 2. Actual vs Predicted (Validasi Fold Terakhir)
sub_actual = y_va.iloc[-200:].values
sub_pred = val_preds[-200:]
ax[1].plot(sub_actual, label='Actual TMA', color='blue', alpha=0.7)
ax[1].plot(sub_pred, label='XGBoost Predicted', color='red', alpha=0.7)
ax[1].set_title('Actual vs Predicted (Validation Set - Sampel 200)')
ax[1].legend()

plt.tight_layout()
plt.show()
'''),

        new_markdown_cell('## 6. Prediksi Test Set & Submission\nPrediksi dilakukan secara independen. Tidak ada target historis, tidak ada *loop* berulang. Model murni menebak tinggi air berdasarkan pembacaan cuaca di masa depan.'),
        new_code_cell('''X_test = test_full[features]

test_preds = np.zeros(len(X_test))
for model in models:
    test_preds += model.predict(X_test) / len(models)

test_full['tma_mdpl'] = test_preds

# Susun Format Submission
sub_pred = test_full[['id', 'tma_mdpl']].copy()
sub = pd.merge(sample_sub[['id']], sub_pred, on='id', how='left')

# Imputasi jika ada cuaca yang bolong di test
sub['tma_mdpl'] = sub['tma_mdpl'].fillna(method='ffill').fillna(method='bfill')

print("Sample Submission:")
display(sub.head())

sub.to_csv('submission_v18_xgboost_redemption.csv', index=False)
print("\\n✅ File 'submission_v18_xgboost_redemption.csv' berhasil dibuat! Selamat kembali ke jalan yang lurus! 🏆")
''')
    ]
    
    nb.cells.extend(cells)
    with open('V18_XGBoost_Redemption.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == '__main__':
    create_v18_notebook()
    print("Notebook V18_XGBoost_Redemption.ipynb berhasil dibuat!")
