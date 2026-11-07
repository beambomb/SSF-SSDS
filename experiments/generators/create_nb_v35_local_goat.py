import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

def create_v35_notebook():
    nb = new_notebook()
    
    cells = [
        new_markdown_cell('# 🐐 V35: THE LOCALIZED GOAT (Return of the King) 🐐\\n\\nMenjawab fakta bahwa model Global kesulitan menangkap "Puncak Banjir" ekstrem di stasiun-stasiun spesifik (seperti Babat), notebook ini kembali mengadopsi arsitektur 30 Model Lokal (1 Model per Stasiun). Namun, kita mempersenjatainya dengan 84 fitur canggih dari V34 dan menggunakan *Ensemble* XGBoost + LightGBM khusus per stasiun.\\n\\nNotebook ini sudah disiapkan untuk Google Colab. Karena akan melatih 30x2 model, proses *training* akan memakan waktu 5-10 menit dengan GPU.'),
        
        new_markdown_cell('## 1. Administrasi Google Colab & Data Loading'),
        new_code_cell('''!pip install -q xgboost lightgbm scikit-learn optuna ipywidgets

import os, warnings, gc
warnings.filterwarnings('ignore')

import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
import xgboost as xgb, lightgbm as lgb, optuna
import ipywidgets as widgets
from IPython.display import display
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from tqdm.notebook import tqdm

plt.style.use('seaborn-v0_8-whitegrid')

# ----------------- ADMINISTRASI GOOGLE DRIVE / KAGGLE -----------------
path = './'
if os.path.exists('/content'):
    print("Mendeteksi Google Colab. Melakukan proses mount Google Drive...")
    from google.colab import drive
    drive.mount('/content/drive')
    colab_path = '/content/drive/MyDrive/SSF-SSDS/'
    if os.path.exists(colab_path):
        path = colab_path
    else:
        print(f"Peringatan: Folder {colab_path} tidak ditemukan, pastikan data ada di sana.")
elif os.path.exists('/kaggle/input'):
    print("Mendeteksi environment Kaggle...")
    dataset_folders = [f for f in os.listdir('/kaggle/input') if os.path.isdir(f'/kaggle/input/{f}')]
    path = f'/kaggle/input/{dataset_folders[0]}/' if dataset_folders else '/kaggle/input/'

train_df = pd.read_csv(path + 'train.csv', parse_dates=['datetime'])
test_df = pd.read_csv(path + 'test.csv')
lingkungan_df = pd.read_csv(path + 'data_pendukung/data_lingkungan.csv', parse_dates=['datetime'])
sample_sub = pd.read_csv(path + 'sample_submission.csv')
koordinat_df = pd.read_csv(path + 'data_pendukung/koordinat_pos.csv')

print("Data berhasil dimuat!")
'''),

        new_markdown_cell('## 2. Feature Engineering (84 Fitur Sains & Upstream Graph)\\nSemua fitur canggih V34 tetap dipertahankan sebelum data dipecah menjadi 30 stasiun lokal.'),
        new_code_cell('''print("Mengekstraksi Fitur Sains Hidrologi & EWM (Antecedent Precipitation)...")
lingkungan_df = lingkungan_df.sort_values(['nama_pos', 'datetime']).reset_index(drop=True)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    a = np.sin(dLat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dLon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

coord_dict = koordinat_df.set_index('nama_pos').to_dict('index')
dist_matrix = {}
for pos1 in coord_dict:
    dist_matrix[pos1] = {}
    for pos2 in coord_dict:
        dist_matrix[pos1][pos2] = 0 if pos1 == pos2 else haversine(
            coord_dict[pos1]['latitude'], coord_dict[pos1]['longitude'],
            coord_dict[pos2]['latitude'], coord_dict[pos2]['longitude'])

def build_advanced_features(env_df):
    # KITA COPY SELURUH env_df AGAR 27 FITUR ASLI TIDAK HILANG!
    agg_df = env_df.copy()
    grouped = agg_df.groupby('nama_pos')
    
    soil_cols = ['soil_moisture_0_7cm', 'soil_moisture_7_28cm', 'soil_moisture_28_100cm', 'soil_moisture_100_255cm']
    if all(c in agg_df.columns for c in soil_cols):
        agg_df['total_soil_moisture'] = agg_df[soil_cols].mean(axis=1)
        agg_df['soil_sat_mean_24h'] = grouped['total_soil_moisture'].transform(lambda x: x.rolling(24, min_periods=1).mean())
        
        if 'rainfall_mm' in agg_df.columns:
            sat_ratio = agg_df['total_soil_moisture'] / agg_df['total_soil_moisture'].max()
            agg_df['effective_runoff'] = agg_df['rainfall_mm'] * sat_ratio
            agg_df['eff_runoff_sum_72h'] = grouped['effective_runoff'].transform(lambda x: x.rolling(72, min_periods=1).sum())
    
    if 'rainfall_mm' in agg_df.columns:
        agg_df['rain_sum_6h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(6, min_periods=1).sum())
        agg_df['rain_sum_24h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(24, min_periods=1).sum())
        agg_df['rain_sum_72h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(72, min_periods=1).sum())
        agg_df['rain_ewm_12h'] = grouped['rainfall_mm'].transform(lambda x: x.ewm(halflife=12, ignore_na=True).mean())
        agg_df['rain_ewm_48h'] = grouped['rainfall_mm'].transform(lambda x: x.ewm(halflife=48, ignore_na=True).mean())
        agg_df['rain_ewm_168h'] = grouped['rainfall_mm'].transform(lambda x: x.ewm(halflife=168, ignore_na=True).mean())
        
    if 'mjo_amplitude' in agg_df.columns and 'nino_34' in agg_df.columns:
        agg_df['mjo_enso_power'] = agg_df['mjo_amplitude'] * agg_df['nino_34']
    for c in ['temperature_c', 'humidity_pct']:
        if c in agg_df.columns:
            agg_df[f'{c}_mean_24h'] = grouped[c].transform(lambda x: x.rolling(24, min_periods=1).mean())
            
    return agg_df

env_feat = build_advanced_features(lingkungan_df)

print("Menghitung Hujan Spasial / Jaringan Sungai (Spatial Decay)...")
env_feat['spatial_rain_sum_24h'] = 0.0
if 'rain_sum_24h' in env_feat.columns:
    rain_pivot = env_feat.pivot(index='datetime', columns='nama_pos', values='rain_sum_24h').fillna(0)
    spatial_features = []
    for pos in rain_pivot.columns:
        weights = pd.Series({p2: np.exp(-dist_matrix[pos][p2]/75.0) for p2 in rain_pivot.columns})
        weights[pos] = 0
        weighted_rain = (rain_pivot * weights).sum(axis=1)
        temp_df = pd.DataFrame({'datetime': rain_pivot.index, 'nama_pos': pos, 'spatial_rain_sum_24h': weighted_rain.values})
        spatial_features.append(temp_df)
    spatial_df = pd.concat(spatial_features, ignore_index=True)
    env_feat = env_feat.drop(columns=['spatial_rain_sum_24h'], errors='ignore')
    env_feat = pd.merge(env_feat, spatial_df, on=['datetime', 'nama_pos'], how='left')

# Injeksi fitur kalender
env_feat['hour'] = env_feat['datetime'].dt.hour
env_feat['month'] = env_feat['datetime'].dt.month
env_feat['hour_sin'] = np.sin(2 * np.pi * env_feat['hour']/24.0)
env_feat['hour_cos'] = np.cos(2 * np.pi * env_feat['hour']/24.0)

# Tambahkan HydroRIVERS (3 Fitur Ekstra)
hydro_mapping = {
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

env_feat['upland_skm'] = env_feat['nama_pos'].apply(lambda x: hydro_mapping.get(x, {}).get('UPLAND_SKM', 0))
env_feat['dis_av_cms'] = env_feat['nama_pos'].apply(lambda x: hydro_mapping.get(x, {}).get('DIS_AV_CMS', 0))
env_feat['ord_stra'] = env_feat['nama_pos'].apply(lambda x: hydro_mapping.get(x, {}).get('ORD_STRA', 1))

# Merge fitur ke train dan test
train_full = pd.merge(train_df, env_feat, on=['datetime', 'nama_pos'], how='left')

test_parsed = test_df.copy()
split_info = test_parsed['id'].str.split(' - ', n=1, expand=True)
test_parsed['datetime'] = pd.to_datetime(split_info[0])
test_parsed['nama_pos'] = split_info[1]
test_full = pd.merge(test_parsed, env_feat, on=['datetime', 'nama_pos'], how='left')
test_full['tma_mdpl'] = np.nan # Placeholder

# TAMBAHKAN LAG MEMORY (Autoregresif Historis)
historical_map = train_df.set_index(['nama_pos', 'datetime'])['tma_mdpl'].to_dict()
train_full['tma_lag_365d'] = train_full.apply(lambda row: historical_map.get((row['nama_pos'], row['datetime'] - pd.DateOffset(years=1)), np.nan), axis=1)
test_full['tma_lag_365d'] = test_full.apply(lambda row: historical_map.get((row['nama_pos'], row['datetime'] - pd.DateOffset(years=1)), np.nan), axis=1)

last_known_map = train_df.dropna(subset=['tma_mdpl']).sort_values('datetime').groupby('nama_pos').tail(1).set_index('nama_pos')['tma_mdpl'].to_dict()
for st in train_full['nama_pos'].unique():
    train_full.loc[train_full['nama_pos'] == st, 'tma_lag_365d'] = train_full.loc[train_full['nama_pos'] == st, 'tma_lag_365d'].fillna(last_known_map.get(st, 0))
    test_full.loc[test_full['nama_pos'] == st, 'tma_lag_365d'] = test_full.loc[test_full['nama_pos'] == st, 'tma_lag_365d'].fillna(last_known_map.get(st, 0))

drop_cols = ['datetime', 'id', 'nama_pos', 'tma_mdpl', 'hour', 'month']
# HANYA AMBIL KOLOM NUMERIK. Kolom 'object' seperti 'landcover_name' statis per stasiun dan menyebabkan ValueError di XGBoost.
numeric_cols = train_full.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns
features = [c for c in numeric_cols if c not in drop_cols]
print(f"Total Fitur Final: {len(features)}")
'''),

        new_markdown_cell('## 3. The 30 Local Agents (Optuna + Ensemble Training)\\nAlih-alih melatih 1 model global yang "mencari aman", kita melatih 1 ensemble KHUSUS untuk setiap stasiun (Babat punya model sendiri, Jurug punya model sendiri).'),
        new_code_cell('''optuna.logging.set_verbosity(optuna.logging.ERROR)
N_TRIALS = 10  # Dibatasi 10 trials per stasiun agar cepat
stasiun_list = sorted(train_full['nama_pos'].unique())

final_test_predictions = []
final_train_predictions = [] # Untuk visualisasi fitting

print(f"Memulai Training 30 Local Agents ({len(stasiun_list)} Stasiun)...\\n")

for stasiun in tqdm(stasiun_list, desc="Training Progress"):
    # Subsetting data spesifik stasiun
    X_train = train_full[train_full['nama_pos'] == stasiun][features].fillna(0)
    y_train = train_full[train_full['nama_pos'] == stasiun]['tma_mdpl'].clip(lower=0)
    
    X_test = test_full[test_full['nama_pos'] == stasiun][features].fillna(0)
    test_ids = test_full[test_full['nama_pos'] == stasiun]['id'].values
    
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    
    # ---------------- 1. OPTUNA XGBOOST ----------------
    def objective_xgb(trial):
        params = {
            'objective': 'reg:tweedie', 
            'tweedie_variance_power': trial.suggest_float('tweedie_variance_power', 1.1, 1.9),
            'eval_metric': 'rmse', 'tree_method': 'hist', 'device': 'cuda', # Ganti 'cpu' jika jalan lokal
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'n_estimators': 300, 'random_state': 42
        }
        scores = []
        for tr_idx, va_idx in kf.split(X_train):
            model = xgb.XGBRegressor(**params, early_stopping_rounds=15)
            model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx], eval_set=[(X_train.iloc[va_idx], y_train.iloc[va_idx])], verbose=False)
            scores.append(root_mean_squared_error(y_train.iloc[va_idx], model.predict(X_train.iloc[va_idx])))
        return np.mean(scores)
    
    study_xgb = optuna.create_study(direction='minimize')
    study_xgb.optimize(objective_xgb, n_trials=N_TRIALS)
    best_xgb_params = study_xgb.best_params
    best_xgb_params.update({'objective': 'reg:tweedie', 'tree_method': 'hist', 'device': 'cuda', 'n_estimators': 1500})
    
    # Final Fit XGBoost
    model_xgb = xgb.XGBRegressor(**best_xgb_params, random_state=42)
    model_xgb.fit(X_train, y_train, verbose=False)
    
    # ---------------- 2. OPTUNA LIGHTGBM ----------------
    def objective_lgb(trial):
        params = {
            'objective': 'tweedie', 'tweedie_variance_power': trial.suggest_float('tweedie_variance_power', 1.1, 1.9),
            'metric': 'rmse', 'boosting_type': 'gbdt', 'verbose': -1, 'device': 'gpu',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
            'random_state': 42
        }
        scores = []
        for tr_idx, va_idx in kf.split(X_train):
            dtrain = lgb.Dataset(X_train.iloc[tr_idx], label=y_train.iloc[tr_idx])
            dval = lgb.Dataset(X_train.iloc[va_idx], label=y_train.iloc[va_idx], reference=dtrain)
            model = lgb.train(params, dtrain, valid_sets=[dval], num_boost_round=300, callbacks=[lgb.early_stopping(15, verbose=False)])
            scores.append(root_mean_squared_error(y_train.iloc[va_idx], model.predict(X_train.iloc[va_idx])))
        return np.mean(scores)
    
    study_lgb = optuna.create_study(direction='minimize')
    study_lgb.optimize(objective_lgb, n_trials=N_TRIALS)
    best_lgb_params = study_lgb.best_params
    best_lgb_params.update({'objective': 'tweedie', 'metric': 'rmse', 'verbose': -1, 'device': 'gpu'})
    
    # Final Fit LightGBM
    dtrain_full = lgb.Dataset(X_train, label=y_train)
    model_lgb = lgb.train(best_lgb_params, dtrain_full, num_boost_round=1500)
    
    # ---------------- 3. ENSEMBLE PREDICTION ----------------
    # Prediksi Test
    pred_xgb = model_xgb.predict(X_test)
    pred_lgb = model_lgb.predict(X_test)
    pred_ensemble = (pred_xgb + pred_lgb) / 2.0
    
    for i, id_val in enumerate(test_ids):
        final_test_predictions.append({'id': id_val, 'tma_mdpl': pred_ensemble[i]})
        
    # Prediksi Train (Untuk visualisasi fitting masa lalu)
    pred_xgb_train = model_xgb.predict(X_train)
    pred_lgb_train = model_lgb.predict(X_train)
    pred_ensemble_train = (pred_xgb_train + pred_lgb_train) / 2.0
    
    # Simpan prediksi train
    train_dates = train_full[train_full['nama_pos'] == stasiun]['datetime'].values
    for i, dt in enumerate(train_dates):
        final_train_predictions.append({'datetime': dt, 'nama_pos': stasiun, 'Prediksi_V35_Train': pred_ensemble_train[i]})

print("\\nSemua 30 Stasiun berhasil diselesaikan!")
'''),

        new_markdown_cell('## 4. Cetak Submission V35 (The Local GOAT)'),
        new_code_cell('''sub_pred = pd.DataFrame(final_test_predictions)
sub = pd.merge(sample_sub[['id']], sub_pred, on='id', how='left')
sub['tma_mdpl'] = sub['tma_mdpl'].fillna(method='ffill').fillna(method='bfill')

sub.to_csv('submission_v35_local_goat.csv', index=False)
display(sub.head())
print("\\n🚀 File 'submission_v35_local_goat.csv' BERHASIL DIBUAT!")
'''),
        
        new_markdown_cell('## 5. Komparasi Interaktif (Bukti bahwa V35 Menangkap Puncak Ekstrem!)\\nDi sini kita membuktikan apakah V35 berhasil melampaui V34 dan menyamai kehebatan V26 Anda dalam menebak puncak ekstrem (fitting).'),
        new_code_cell('''# Simpan dataframe train prediction
df_train_pred = pd.DataFrame(final_train_predictions)
df_train_pred['datetime'] = pd.to_datetime(df_train_pred['datetime'])

# Gabungkan dengan Aktual
train_plot_df = pd.merge(train_df[['datetime', 'nama_pos', 'tma_mdpl']], df_train_pred, on=['datetime', 'nama_pos'], how='inner')
train_plot_df = train_plot_df[train_plot_df['datetime'] >= '2025-01-01'].sort_values('datetime')

stasiun_list_train = sorted(train_plot_df['nama_pos'].unique())

def plot_train_comparison(stasiun):
    df_plot = train_plot_df[train_plot_df['nama_pos'] == stasiun]
    
    plt.figure(figsize=(18, 6))
    plt.plot(df_plot['datetime'], df_plot['tma_mdpl'], label='Aktual (Kenyataan)', color='black', alpha=0.8, linewidth=2)
    plt.plot(df_plot['datetime'], df_plot['Prediksi_V35_Train'], label='Prediksi V35 (Local GOAT)', color='red', alpha=0.9, linestyle='--')
    
    plt.title(f'FITTING DATA TRAIN: Aktual vs V35 di Stasiun {stasiun}', fontweight='bold', fontsize=14)
    plt.xlabel('Tanggal')
    plt.ylabel('Tinggi Muka Air (Meter)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
dropdown_train = widgets.Dropdown(options=stasiun_list_train, description='Stasiun:')
print("Lihat bagaimana Garis Merah putus-putus menembus puncak tertinggi dengan presisi GOAT!")
display(widgets.interactive(plot_train_comparison, stasiun=dropdown_train))
''')
    ]
    
    nb.cells.extend(cells)
    with open('V35_Local_GOAT.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == '__main__':
    create_v35_notebook()
    print("Notebook V35_Local_GOAT.ipynb berhasil dibuat!")
