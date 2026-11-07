import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

def create_v21_notebook():
    nb = new_notebook()
    
    cells = [
        new_markdown_cell('# 🌍 V21 LOCAL MODELING (30 AI SPESIALIS) 🌍\nBerdasarkan investigasi kita yang mendalam, performa model memburuk saat dijejali fitur rumit (V19/V20) karena AI "mabuk" mencoba mencari satu rumus ajaib untuk 30 sungai yang berbeda secara geografis.\nDi V21 ini, kita melakukan REVOLUSI: **Melatih 30 Model XGBoost yang Berbeda!** Setiap stasiun akan memiliki AI pribadinya sendiri. Dengan begini, AI dipaksa untuk 100% menggunakan fitur cuaca (Juara V18) dan tidak lagi menjadikan elevasi sebagai jalan pintas!'),
        
        new_markdown_cell('## 1. Persiapan Environment & Pemuatan Data'),
        new_code_cell('''!pip install -q xgboost scikit-learn
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
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

        new_markdown_cell('## 2. Feature Engineering (Kembali ke Fitur Juara V18)\nKita membuang fitur-fitur kompleks yang membuat model global mabuk. Kita menggunakan formula paling bersih yang telah membuktikan diri mencetak skor 1.64!'),
        new_code_cell('''print("Mengekstraksi Fitur Cuaca (V18 murni)...")
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
    
    # KITA TIDAK LAGI MEMAKAI KOORDINAT DAN ELEVASI!
    # Kenapa? Karena setiap model hanya melihat 1 stasiun, nilai tersebut konstan dan 100% tidak berguna!
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
print(f"Total Fitur Cuaca Murni: {len(features)}")
'''),

        new_markdown_cell('## 3. Training 30 Model Spesialis (Local Modeling) 🧠\nDaripada menyiksa Optuna untuk 1 model lambat, kita langsung menggunakan parameter Optuna terbaik dari V18 dan melatih **30 XGBoost berbeda** (satu per stasiun). Waktu komputasi total akan sangat cepat karena data dipecah 30!'),
        new_code_cell('''# Parameter Juara V18
best_params = {
    'learning_rate': 0.03, 'max_depth': 7, 'subsample': 0.8, 
    'colsample_bytree': 0.9, 'min_child_weight': 3, 'reg_lambda': 1.0,
    'objective': 'reg:squarederror', 'eval_metric': 'rmse', 
    'tree_method': 'hist', 'n_estimators': 1500, 'random_state': 42
}

stations = train_full['nama_pos'].unique()
station_models = {}
station_scores = {}

print("Memulai Pelatihan 30 Model AI Spesialis...")

for st in stations:
    # Ambil data spesifik hanya untuk stasiun ini
    X_st = train_full[train_full['nama_pos'] == st][features]
    y_st = train_full[train_full['nama_pos'] == st]['tma_mdpl']
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    models = []
    scores = []
    
    for train_idx, val_idx in kf.split(X_st, y_st):
        X_tr, y_tr = X_st.iloc[train_idx], y_st.iloc[train_idx]
        X_va, y_va = X_st.iloc[val_idx], y_st.iloc[val_idx]
        
        model = xgb.XGBRegressor(**best_params, early_stopping_rounds=50)
        # Verbose False agar tidak memenuhi layar
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        
        val_preds = model.predict(X_va)
        scores.append(root_mean_squared_error(y_va, val_preds))
        models.append(model)
        
    avg_rmse = np.mean(scores)
    station_models[st] = models
    station_scores[st] = avg_rmse
    print(f"✅ Model Spesialis [{st}] - Avg CV RMSE: {avg_rmse:.4f}")

global_rmse = np.mean(list(station_scores.values()))
print(f"\\n🏆 GRAND AVERAGE RMSE ACROSS 30 STATIONS: {global_rmse:.4f}")
'''),

        new_markdown_cell('## 4. Visualisasi Sejati Per Pos 🔍\nSekarang kita benar-benar melihat *Feature Importance* tanpa gangguan elevasi! Model DIPAKSA menganalisis cuaca!'),
        new_code_cell('''# Pilih Stasiun untuk divisualisasikan (Bisa diganti-ganti!)
TARGET_STATION = stations[0]

fig, ax = plt.subplots(1, 2, figsize=(22, 8))

# 1. Feature Importance Murni Cuaca
feat_imp = np.mean([m.feature_importances_ for m in station_models[TARGET_STATION]], axis=0)
imp_df = pd.DataFrame({'Features': features, 'Importance score': feat_imp})
imp_df = imp_df.sort_values(by='Importance score', ascending=False).head(20)

sns.barplot(x='Importance score', y='Features', data=imp_df, ax=ax[0], palette='magma')
ax[0].set_title(f'Top 20 Feature Importance Spesialis [{TARGET_STATION}]')

# 2. Actual vs Predicted (Validasi)
# Ambil sebagian data validasi dari stasiun target
X_target = train_full[train_full['nama_pos'] == TARGET_STATION][features]
y_target = train_full[train_full['nama_pos'] == TARGET_STATION]['tma_mdpl']

val_preds = np.zeros(len(X_target))
for m in station_models[TARGET_STATION]:
    val_preds += m.predict(X_target) / 5

# Plot 200 hari terakhir dari stasiun ini
sub_actual = y_target.iloc[-200:].values
sub_pred = val_preds[-200:]
ax[1].plot(sub_actual, label='Actual TMA', color='blue', alpha=0.7, linewidth=2)
ax[1].plot(sub_pred, label='XGBoost Predicted', color='red', alpha=0.7, linewidth=2)
ax[1].set_title(f'Actual vs Predicted - 200 Hari Terakhir [{TARGET_STATION}]')
ax[1].legend()

plt.tight_layout()
plt.show()
'''),

        new_markdown_cell('## 5. Prediksi Test Set Per Pos\nMasing-masing stasiun pada Test Set ditebak oleh AI spesialisnya sendiri.'),
        new_code_cell('''test_preds = np.zeros(len(test_full))

for st in stations:
    # Temukan index baris test untuk stasiun ini
    idx = test_full['nama_pos'] == st
    if idx.sum() == 0: continue
        
    X_test_st = test_full.loc[idx, features]
    
    st_preds = np.zeros(len(X_test_st))
    for m in station_models[st]:
        st_preds += m.predict(X_test_st) / 5
        
    test_full.loc[idx, 'tma_mdpl'] = st_preds

sub_pred = test_full[['id', 'tma_mdpl']].copy()
sub = pd.merge(sample_sub[['id']], sub_pred, on='id', how='left')

# Imputasi Missing
sub['tma_mdpl'] = sub['tma_mdpl'].fillna(method='ffill').fillna(method='bfill')

print("Sample Submission:")
display(sub.head())

sub.to_csv('submission_v21_local_models.csv', index=False)
print("\\n✅ File 'submission_v21_local_models.csv' berhasil dibuat dengan 30 AI Spesialis! Taklukkan Kaggle!")
''')
    ]
    
    nb.cells.extend(cells)
    with open('V21_XGBoost_Local_Models.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == '__main__':
    create_v21_notebook()
    print("Notebook V21_XGBoost_Local_Models.ipynb berhasil dibuat!")
