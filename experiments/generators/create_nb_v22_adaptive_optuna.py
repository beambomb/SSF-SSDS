import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

def create_v22_notebook():
    nb = new_notebook()
    
    cells = [
        new_markdown_cell('# 🧬 V22 ADAPTIVE OPTUNA LOCAL MODELS 🧬\nDi V21, kita menyadari bahwa stasiun seperti "Bojonegoro - Kali Kethek" sangat fluktuatif (kemungkinan karena operasi pintu air atau pasang surut) sehingga AI yang di-set "main aman" menolak memprediksinya. Di V22 ini, kita membiarkan **Optuna berjalan secara independen untuk setiap stasiun**. Optuna akan mengizinkan pohon yang sangat dalam untuk stasiun yang liar, dan pohon yang dangkal untuk stasiun yang tenang. 30 AI Spesialis Sejati telah lahir!'),
        
        new_markdown_cell('## 1. Persiapan Environment & Pemuatan Data'),
        new_code_cell('''!pip install -q xgboost scikit-learn optuna
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

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

        new_markdown_cell('## 2. Feature Engineering (Evolusi Fitur Siklus Harian)\nSelain fitur V18, saya menambahkan `hour_sin_12h` dan `hour_cos_12h` untuk menangkap siklus pasang surut laut (yang umumnya memiliki periode ~12 jam). Ini sangat krusial untuk stasiun liar seperti Bojonegoro!'),
        new_code_cell('''print("Mengekstraksi Fitur Cuaca & Pasang Surut...")
lingkungan_df = lingkungan_df.sort_values(['nama_pos', 'datetime']).reset_index(drop=True)

def build_exogenous_features(env_df):
    agg_df = env_df[['datetime', 'nama_pos']].copy()
    grouped = env_df.groupby('nama_pos')
    
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

    return agg_df

env_feat = build_exogenous_features(lingkungan_df)

def prepare_dataset(base_df, env_feat):
    df = base_df.copy()
    
    # Waktu Harian (24 Jam)
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['dayofyear'] = df['datetime'].dt.dayofyear
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12.0)
    
    # Siklus Pasang Surut / Operasi Pintu Air (12 Jam)
    df['hour_sin_12h'] = np.sin(2 * np.pi * df['hour']/12.0)
    df['hour_cos_12h'] = np.cos(2 * np.pi * df['hour']/12.0)
    
    # Merge Lingkungan
    df = pd.merge(df, env_feat, on=['datetime', 'nama_pos'], how='left')
    
    return df

print("Menyusun Train Set...")
train_full = prepare_dataset(train_df, env_feat)

print("Menyusun Test Set...")
test_parsed = test_df.copy()
split_info = test_parsed['id'].str.split(' - ', n=1, expand=True)
test_parsed['datetime'] = pd.to_datetime(split_info[0])
test_parsed['nama_pos'] = split_info[1]
test_full = prepare_dataset(test_parsed, env_feat)

train_full = train_full.dropna(subset=['tma_mdpl']).reset_index(drop=True)

drop_cols = ['datetime', 'nama_pos', 'tma_mdpl', 'id']
features = [c for c in train_full.columns if c not in drop_cols]
print(f"Total Fitur Cuaca & Pasang Surut: {len(features)}")
'''),

        new_markdown_cell('## 3. Adaptive Optuna & Training (30 Model)\nBagian ini mungkin akan memakan waktu 5-10 menit. Optuna akan berjalan senyap untuk SETIAP stasiun secara bergantian. Stasiun yang tenang akan diberikan pohon yang dangkal (mencegah *overfit*), stasiun yang liar akan diizinkan tumbuh pohon yang dalam untuk menghafal pola pintu air!'),
        new_code_cell('''stations = train_full['nama_pos'].unique()
station_models = {}
station_scores = {}

# Matikan log Optuna agar terminal tidak penuh
optuna.logging.set_verbosity(optuna.logging.ERROR)

N_TRIALS = 15 # Mencari parameter 15 kali per stasiun

print(f"Memulai Evolusi Optuna untuk 30 Stasiun (Total {30 * N_TRIALS} Trials)...\\n")

for st in stations:
    X_st = train_full[train_full['nama_pos'] == st][features]
    y_st = train_full[train_full['nama_pos'] == st]['tma_mdpl']
    
    # --- 1. OPTUNA PER STASIUN ---
    def objective(trial):
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'hist',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12), # Diperlebar sampai 12 untuk stasiun liar
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'n_estimators': 800, # Dikurangi sedikit agar komputasi 30 stasiun tetap cepat
            'random_state': 42
        }
        
        # Validasi 80-20 untuk Optuna
        split_idx = int(len(X_st) * 0.8)
        X_tr, y_tr = X_st.iloc[:split_idx], y_st.iloc[:split_idx]
        X_va, y_va = X_st.iloc[split_idx:], y_st.iloc[split_idx:]
        
        model = xgb.XGBRegressor(**params, early_stopping_rounds=30)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        
        preds = model.predict(X_va)
        return root_mean_squared_error(y_va, preds)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS)
    
    best_params = study.best_params
    best_params['objective'] = 'reg:squarederror'
    best_params['eval_metric'] = 'rmse'
    best_params['tree_method'] = 'hist'
    best_params['n_estimators'] = 800
    best_params['random_state'] = 42

    # --- 2. K-FOLD TRAINING DENGAN BEST PARAMS ---
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
    
    # Berikan warning jika stasiun memiliki error di atas 1.0 (Stasiun Liar)
    status = "⚠️ LIAR" if avg_rmse > 1.0 else "✅ AMAN"
    print(f"{status} [{st}] - Depth: {best_params['max_depth']} | RMSE: {avg_rmse:.4f}")

global_rmse = np.mean(list(station_scores.values()))
print(f"\\n🏆 GRAND AVERAGE RMSE ACROSS 30 STATIONS: {global_rmse:.4f}")
'''),

        new_markdown_cell('## 4. Visualisasi Spesifik\nCoba ganti `TARGET_STATION` ke "Bojonegoro - Kali Kethek" dan perhatikan apakah AI sekarang bisa menangkap pola pasang surutnya!'),
        new_code_cell('''# Pilih Stasiun
TARGET_STATION = "Bojonegoro - Kali Kethek" # Ganti dengan nama stasiun yang ingin dicek

fig, ax = plt.subplots(1, 2, figsize=(22, 8))

# 1. Feature Importance
feat_imp = np.mean([m.feature_importances_ for m in station_models[TARGET_STATION]], axis=0)
imp_df = pd.DataFrame({'Features': features, 'Importance score': feat_imp})
imp_df = imp_df.sort_values(by='Importance score', ascending=False).head(20)

sns.barplot(x='Importance score', y='Features', data=imp_df, ax=ax[0], palette='viridis')
ax[0].set_title(f'Top 20 Feature Importance [{TARGET_STATION}]')

# 2. Actual vs Predicted (Validasi)
X_target = train_full[train_full['nama_pos'] == TARGET_STATION][features]
y_target = train_full[train_full['nama_pos'] == TARGET_STATION]['tma_mdpl']

val_preds = np.zeros(len(X_target))
for m in station_models[TARGET_STATION]:
    val_preds += m.predict(X_target) / 5

sub_actual = y_target.iloc[-200:].values
sub_pred = val_preds[-200:]
ax[1].plot(sub_actual, label='Actual TMA', color='blue', alpha=0.7, linewidth=2)
ax[1].plot(sub_pred, label='XGBoost Predicted', color='red', alpha=0.7, linewidth=2)
ax[1].set_title(f'Actual vs Predicted - 200 Hari Terakhir [{TARGET_STATION}]')
ax[1].legend()

plt.tight_layout()
plt.show()
'''),

        new_markdown_cell('## 5. Prediksi Test Set\nHasil akhir yang diprediksi oleh 30 AI Spesialis (Zero Data Leakage).'),
        new_code_cell('''test_preds = np.zeros(len(test_full))

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

print("Sample Submission:")
display(sub.head())

sub.to_csv('submission_v22_adaptive_optuna.csv', index=False)
print("\\n✅ File 'submission_v22_adaptive_optuna.csv' berhasil dibuat dengan kekuatan Adaptive Optuna! Taklukkan Kaggle!")
''')
    ]
    
    nb.cells.extend(cells)
    with open('V22_Adaptive_Optuna.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == '__main__':
    create_v22_notebook()
    print("Notebook V22_Adaptive_Optuna.ipynb berhasil dibuat!")
