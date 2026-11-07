import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

def create_v26_5_notebook():
    nb = new_notebook()
    
    cells = [
        new_markdown_cell('# 🗺️ V26.5 TOPOLOGICAL XGBOOST (Tanpa Target Leakage) 🗺️\\nBerdasarkan skor 1.61 Anda di V26, kita menambahkan arsitektur *Spatial/Graph* ke dalam XGBoost ini.\\nKodenya akan otomatis memetakan pos mana yang mengalir ke mana (Hulu ke Hilir), lalu menyuntikkan data *Upstream* (seperti curah hujan hulu dan kapasitas bendungan hulu) ke dataset Hilir.\\n\\nIni memungkinkan XGBoost menyimulasikan pergerakan air sungai secara presisi tanpa melanggar aturan *Target Leakage*!'),
        
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
from sklearn.model_selection import KFold
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
coords_df = pd.read_csv(path + 'data_pendukung/koordinat_pos.csv')

print("Data berhasil dimuat!")
'''),

        new_markdown_cell('## 2. Pemetaan Topologi Otomatis (The River Graph)'),
        new_code_cell('''print("Membangun Peta Topologi Sungai...")

# 1. Hitung Elevasi rata-rata tiap pos untuk menentukan Hulu/Hilir
elevations = train_df.groupby('nama_pos')['tma_mdpl'].mean().to_dict()
coords = coords_df.copy()
coords['elevation'] = coords['nama_pos'].map(elevations)

upstream_map = {pos: [] for pos in coords['nama_pos']}

# 2. Algoritma Pencarian Hilir Terdekat
for i, row_i in coords.iterrows():
    pos_i = row_i['nama_pos']
    elev_i = row_i['elevation']
    
    # Cari kandidat hilir (elevasi lebih rendah dari pos ini)
    candidates = coords[coords['elevation'] < elev_i].copy()
    if len(candidates) > 0:
        # Hitung jarak Euclidean
        dist = np.sqrt((candidates['latitude'] - row_i['latitude'])**2 + (candidates['longitude'] - row_i['longitude'])**2)
        nearest_idx = dist.idxmin()
        pos_hilir = coords.loc[nearest_idx, 'nama_pos']
        
        # Jika pos_i mengalir ke pos_hilir, maka bagi pos_hilir, pos_i adalah HULU (Upstream)
        upstream_map[pos_hilir].append(pos_i)

# Tampilkan beberapa pemetaan
print("\\nContoh Pemetaan Aliran:")
for pos in list(upstream_map.keys())[:5]:
    hulu = upstream_map[pos]
    if len(hulu) > 0:
        print(f"[{pos}] menerima air dari Hulu: {hulu}")
    else:
        print(f"[{pos}] adalah titik paling Hulu (tidak menerima kiriman).")
'''),

        new_markdown_cell('## 3. Advanced Feature Engineering (Lokal, Basin, & Topological Upstream)'),
        new_code_cell('''print("\\nMengekstraksi Fitur Cuaca Lokal & GLOBAL BASIN...")
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
        
    # FITUR GLOBAL BASIN (Helicopter View)
    basin_env = env_df.groupby('datetime').agg({'rainfall_mm': 'mean', 'soil_moisture_0_7cm': 'mean'}).reset_index()
    basin_env['basin_rain_sum_72h'] = basin_env['rainfall_mm'].rolling(72, min_periods=1).sum()
    basin_env['basin_rain_sum_168h'] = basin_env['rainfall_mm'].rolling(168, min_periods=1).sum()
    basin_env['basin_rain_sum_360h'] = basin_env['rainfall_mm'].rolling(360, min_periods=1).sum()
    basin_env['basin_soil_mean_168h'] = basin_env['soil_moisture_0_7cm'].rolling(168, min_periods=1).mean()
    
    basin_features = basin_env[['datetime', 'basin_rain_sum_72h', 'basin_rain_sum_168h', 'basin_rain_sum_360h', 'basin_soil_mean_168h']]
    agg_df = pd.merge(agg_df, basin_features, on='datetime', how='left')

    return agg_df

env_feat = build_exogenous_features(lingkungan_df)

def prepare_dataset_topological(base_df, env_feat, historical_tma_df, upstream_map):
    df = base_df.copy()
    
    # 📌 TAHAP 1: FITUR DASAR & JANGKAR V26
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
    
    # Jangkar 1: Memori Tahun Lalu
    df['datetime_minus_1_year'] = df['datetime'] - pd.DateOffset(years=1)
    historical_map = historical_tma_df.set_index(['nama_pos', 'datetime'])['tma_mdpl'].to_dict()
    df['tma_lag_365d'] = df.apply(lambda row: historical_map.get((row['nama_pos'], row['datetime_minus_1_year']), np.nan), axis=1)
    
    # Jangkar 2: Last Known TMA
    last_known_map = historical_tma_df.dropna(subset=['tma_mdpl']).sort_values('datetime').groupby('nama_pos').tail(1).set_index('nama_pos')['tma_mdpl'].to_dict()
    df['last_known_tma'] = df['nama_pos'].map(last_known_map)
    df = df.drop(columns=['datetime_minus_1_year'])
    
    # 📌 TAHAP 2: INJEKSI TOPOLOGI HULU (UPSTREAM FEATURES)
    # Kita membuat DataFrame fitur hulu secara terpisah lalu di-merge agar efisien (Vectorized)
    print("  -> Menyuntikkan Fitur Aliran Sungai Hulu (Upstream)...")
    upstream_dfs = []
    
    for pos, upstreams in upstream_map.items():
        if len(upstreams) > 0:
            # Filter baris yang merupakan Hulu dari pos ini
            up_df = df[df['nama_pos'].isin(upstreams)]
            # Rata-ratakan nilai curah hujan dan jangkar dari Hulu pada waktu yang sama
            agg = up_df.groupby('datetime')[['rain_sum_72h', 'last_known_tma', 'tma_lag_365d']].mean().reset_index()
        else:
            # Jika pos ini berada di ujung hulu pegunungan (tidak ada kiriman), 
            # gunakan nilainya sendiri agar array tetap lengkap.
            up_df = df[df['nama_pos'] == pos]
            agg = up_df[['datetime', 'rain_sum_72h', 'last_known_tma', 'tma_lag_365d']].copy()
        
        # Ganti nama kolom untuk menandakan ini adalah data kiriman dari Hulu
        agg = agg.rename(columns={
            'rain_sum_72h': 'upstream_rain_sum_72h',
            'last_known_tma': 'upstream_last_known_tma',
            'tma_lag_365d': 'upstream_tma_lag_365d'
        })
        agg['nama_pos'] = pos
        upstream_dfs.append(agg)
        
    all_upstream = pd.concat(upstream_dfs, ignore_index=True)
    
    # Gabungkan fitur hulu kembali ke dataset asli
    df = pd.merge(df, all_upstream, on=['datetime', 'nama_pos'], how='left')
    
    return df

print("Menyusun Train & Test Set dengan Topological Features...")
historical_tma = train_df[['datetime', 'nama_pos', 'tma_mdpl']].copy()

train_full = prepare_dataset_topological(train_df, env_feat, historical_tma, upstream_map)

test_parsed = test_df.copy()
split_info = test_parsed['id'].str.split(' - ', n=1, expand=True)
test_parsed['datetime'] = pd.to_datetime(split_info[0])
test_parsed['nama_pos'] = split_info[1]
test_full = prepare_dataset_topological(test_parsed, env_feat, historical_tma, upstream_map)

# Forward/Backward fill untuk NaN pada Jangkar Memori
for st in train_full['nama_pos'].unique():
    mask_tr = train_full['nama_pos'] == st
    train_full.loc[mask_tr, 'tma_lag_365d'] = train_full.loc[mask_tr, 'tma_lag_365d'].fillna(train_full.loc[mask_tr, 'last_known_tma'])
    train_full.loc[mask_tr, 'upstream_tma_lag_365d'] = train_full.loc[mask_tr, 'upstream_tma_lag_365d'].fillna(train_full.loc[mask_tr, 'upstream_last_known_tma'])
    
    mask_te = test_full['nama_pos'] == st
    test_full.loc[mask_te, 'tma_lag_365d'] = test_full.loc[mask_te, 'tma_lag_365d'].fillna(test_full.loc[mask_te, 'last_known_tma'])
    test_full.loc[mask_te, 'upstream_tma_lag_365d'] = test_full.loc[mask_te, 'upstream_tma_lag_365d'].fillna(test_full.loc[mask_te, 'upstream_last_known_tma'])

train_full = train_full.dropna(subset=['tma_mdpl']).reset_index(drop=True)

drop_cols = ['datetime', 'nama_pos', 'tma_mdpl', 'id']
features = [c for c in train_full.columns if c not in drop_cols]
print(f"Total Fitur: {len(features)}")
print(f"Daftar Fitur: {features}")
'''),

        new_markdown_cell('## 4. Optuna: Tweedie Loss dengan Kekuatan Spasial (Graph)\\nXGBoost kini otomatis mengecek kondisi bendungan Hulu sebelum menebak air Hilir!'),
        new_code_cell('''stations = train_full['nama_pos'].unique()
station_models = {}
station_scores = {}

optuna.logging.set_verbosity(optuna.logging.ERROR)
N_TRIALS = 15

print(f"Memulai Optuna (Tweedie) untuk 30 Stasiun (Total {30 * N_TRIALS} Trials)...\\n")

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
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'n_estimators': 800,
            'random_state': 42
        }
        
        split_idx = int(len(X_st) * 0.8)
        X_tr, y_tr = X_st.iloc[:split_idx], np.clip(y_st.iloc[:split_idx], 0.0001, None)
        X_va, y_va = X_st.iloc[split_idx:], np.clip(y_st.iloc[split_idx:], 0.0001, None)
        
        model = xgb.XGBRegressor(**params, early_stopping_rounds=30)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        
        preds = model.predict(X_va)
        return root_mean_squared_error(y_va, preds)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS)
    
    best_params = study.best_params
    best_params['objective'] = 'reg:tweedie'
    best_params['eval_metric'] = 'rmse'
    best_params['tree_method'] = 'hist'
    best_params['n_estimators'] = 800
    best_params['random_state'] = 42

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    models = []
    scores = []
    
    for train_idx, val_idx in kf.split(X_st, y_st):
        X_tr, y_tr = X_st.iloc[train_idx], np.clip(y_st.iloc[train_idx], 0.0001, None)
        X_va, y_va = X_st.iloc[val_idx], np.clip(y_st.iloc[val_idx], 0.0001, None)
        
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
        new_code_cell('''TARGET_STATION = "Jurug" # Coba cek stasiun Hilir!

fig, ax = plt.subplots(1, 2, figsize=(22, 8))

feat_imp = np.mean([m.feature_importances_ for m in station_models[TARGET_STATION]], axis=0)
imp_df = pd.DataFrame({'Features': features, 'Importance score': feat_imp})
imp_df = imp_df.sort_values(by='Importance score', ascending=False).head(20)

sns.barplot(x='Importance score', y='Features', data=imp_df, ax=ax[0], palette='viridis')
ax[0].set_title(f'Top 20 Feature Importance [{TARGET_STATION}]')

X_target = train_full[train_full['nama_pos'] == TARGET_STATION][features]
y_target = train_full[train_full['nama_pos'] == TARGET_STATION]['tma_mdpl']

val_preds = np.zeros(len(X_target))
for m in station_models[TARGET_STATION]:
    val_preds += m.predict(X_target) / 5

sub_actual = y_target.iloc[-200:].values
sub_pred = val_preds[-200:]
ax[1].plot(sub_actual, label='Actual TMA', color='blue', alpha=0.7, linewidth=2)
ax[1].plot(sub_pred, label='Predicted (Topological XGBoost)', color='red', alpha=0.7, linewidth=2)
ax[1].set_title(f'Actual vs Predicted - 200 Hari Terakhir [{TARGET_STATION}]')
ax[1].legend()

plt.tight_layout()
plt.show()

# ================= BAGIAN TEST SET =================
test_preds = np.zeros(len(test_full))

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
sub.to_csv('submission_v26_5_topological_xgb.csv', index=False)
print("\\n✅ File 'submission_v26_5_topological_xgb.csv' berhasil dibuat dengan kekuatan Topological Graph!")
''')
    ]
    
    nb.cells.extend(cells)
    with open('V26_5_Topological_XGBoost.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == '__main__':
    create_v26_5_notebook()
    print("Notebook V26_5_Topological_XGBoost.ipynb berhasil dibuat!")
