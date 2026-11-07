import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# === Cell 1: Setup & Import ===
cell_setup = nbf.v4.new_code_cell("""# === TAHAP 1: PERSIAPAN & ADMINISTRASI KAGGLE/COLAB ===
import os
import gc
import warnings
warnings.filterwarnings('ignore')

IN_COLAB = False
try:
    import google.colab
    IN_COLAB = True
    print("Berjalan di Google Colab. Melakukan mount drive...")
    from google.colab import drive
    drive.mount('/content/drive')
except:
    print("Tidak di Colab. Menggunakan environment lokal/Kaggle.")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import math

!pip install -q catboost optuna statsmodels
import optuna
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import acf

# Sesuaikan path ini dengan struktur folder Drive Anda!
if IN_COLAB:
    DATA_DIR = '/content/drive/MyDrive/Kaggle/SSF-SSDS/data_pendukung'
    TRAIN_PATH = '/content/drive/MyDrive/Kaggle/SSF-SSDS/train.csv'
    TEST_PATH = '/content/drive/MyDrive/Kaggle/SSF-SSDS/test.csv'
    SAMPLE_SUB_PATH = '/content/drive/MyDrive/Kaggle/SSF-SSDS/sample_submission.csv'
else:
    # Fallback untuk Kaggle Environment atau Local Workspace
    DATA_DIR = './data_pendukung'
    TRAIN_PATH = './train.csv'
    TEST_PATH = './test.csv'
    SAMPLE_SUB_PATH = './sample_submission.csv'

# Verifikasi
if not os.path.exists(TRAIN_PATH):
    print("⚠️ WARNING: Jalur dataset tidak ditemukan. Harap pastikan folder Kaggle di Drive Anda sudah benar.")
else:
    print("✅ Lingkungan tempur siap!")""")

# === Cell 2: Load Data ===
cell_load = nbf.v4.new_code_cell("""# === TAHAP 2: LOAD DATA ===
print("Memuat data utama...")
train_df = pd.read_csv(TRAIN_PATH, parse_dates=['datetime'])
test_df = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
lingkungan_df = pd.read_csv(os.path.join(DATA_DIR, 'data_lingkungan.csv'), parse_dates=['datetime'])

print(f"Train Shape: {train_df.shape}")
print(f"Test Shape: {test_df.shape}")""")

# === Cell 3: Sensor Exorcism ===
cell_exorcism = nbf.v4.new_code_cell("""# === TAHAP 3: SENSOR EXORCISM (MAD FILTER) ===
def detect_and_clean_glitches(df, target_col='tma_mdpl', threshold=4.0):
    cleaned_df = df.copy()
    stations = cleaned_df['nama_pos'].unique()
    total_glitches = 0
    
    for st in stations:
        mask = cleaned_df['nama_pos'] == st
        st_data = cleaned_df.loc[mask].copy()
        
        # Hitung lonjakan antar baris (Forward & Backward)
        diff_f = st_data[target_col].diff().abs()
        diff_b = st_data[target_col].diff(-1).abs()
        
        # Median Absolute Deviation (MAD) dari lonjakan
        mad_f = diff_f.median()
        mad_b = diff_b.median()
        
        # Jika MAD terlalu kecil, pakai threshold minimum agar tidak over-sensitive
        mad_f = max(mad_f, 0.1)
        mad_b = max(mad_b, 0.1)
        
        # Deteksi: Naik tajam DAN turun tajam seketika (1-point spike)
        is_spike = (diff_f > threshold * mad_f) & (diff_b > threshold * mad_b)
        
        # Interpolasi baris yang terkena glitch
        st_data.loc[is_spike, target_col] = np.nan
        st_data[target_col] = st_data[target_col].interpolate(method='linear')
        
        cleaned_df.loc[mask, target_col] = st_data[target_col]
        total_glitches += is_spike.sum()
        
    print(f"🔮 [EXORCISM] Ditemukan dan dibersihkan {total_glitches} baris data hantu (Glitch)!")
    return cleaned_df

train_clean = detect_and_clean_glitches(train_df)
""")

# === Cell 4: Baseflow ===
cell_baseflow = nbf.v4.new_code_cell("""# === TAHAP 4: FISIKA HIDROLOGI (LYNE-HOLLICK BASEFLOW) ===
def apply_lyne_hollick(df, rain_col='rainfall_mm', alpha=0.925):
    '''Memisahkan hujan menjadi Quickflow (Banjir instan) dan Baseflow (Serapan tanah)'''
    # Asumsikan df sudah diurutkan berdasarkan waktu
    rain = df[rain_col].fillna(0).values
    qf = np.zeros_like(rain)
    bf = np.zeros_like(rain)
    
    bf[0] = rain[0] * 0.5 # Inisialisasi tebakan
    for t in range(1, len(rain)):
        # Rumus standar Lyne-Hollick (Low Pass Filter)
        bf[t] = alpha * bf[t-1] + ((1 - alpha) / 2) * (rain[t] + rain[t-1])
        # Baseflow tidak boleh lebih dari total hujan
        bf[t] = min(bf[t], rain[t]) if rain[t] > 0 else max(bf[t], 0)
        
    qf = rain - bf
    qf = np.where(qf < 0, 0, qf) # Tidak boleh negatif
    return qf, bf

lingkungan_df = lingkungan_df.sort_values(['nama_pos', 'datetime']).reset_index(drop=True)

qf_all, bf_all = [], []
for st in lingkungan_df['nama_pos'].unique():
    mask = lingkungan_df['nama_pos'] == st
    qf, bf = apply_lyne_hollick(lingkungan_df[mask])
    lingkungan_df.loc[mask, 'quickflow_mm'] = qf
    lingkungan_df.loc[mask, 'baseflow_mm'] = bf

# Akumulasi dinamis untuk menangkap delay
lingkungan_df['qf_sum_24h'] = lingkungan_df.groupby('nama_pos')['quickflow_mm'].transform(lambda x: x.rolling(24, min_periods=1).sum())
lingkungan_df['bf_sum_168h'] = lingkungan_df.groupby('nama_pos')['baseflow_mm'].transform(lambda x: x.rolling(168, min_periods=1).sum())

print("Pemisahan Lyne-Hollick Selesai!")""")

# === Cell 5: Pure Forcing Fusion ===
cell_fusion = nbf.v4.new_code_cell("""# === TAHAP 5: PURE FORCING FUSION ===
def prepare_pure_forcing(base_df, env_df):
    df = base_df.copy()
    
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['dayofyear'] = df['datetime'].dt.dayofyear
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24.0)
    
    df = pd.merge(df, env_df, on=['datetime', 'nama_pos'], how='left')
    df = df.sort_values(['nama_pos', 'datetime']).reset_index(drop=True)
    return df

print("Menyusun Train Set (Pure Forcing)...")
train_full = prepare_pure_forcing(train_clean, lingkungan_df)

print("Menyusun Test Set (Pure Forcing)...")
test_parsed = test_df.copy()
split_info = test_parsed['id'].str.split(' - ', n=1, expand=True)
test_parsed['datetime'] = pd.to_datetime(split_info[0])
test_parsed['nama_pos'] = split_info[1]
test_full = prepare_pure_forcing(test_parsed, lingkungan_df)

# Drop baris train yang targetnya bolong
train_full = train_full.dropna(subset=['tma_mdpl']).reset_index(drop=True)

drop_cols = ['datetime', 'tma_mdpl', 'tma_asli', 'id'] # 'nama_pos' JANGAN DIDROP! Dibutuhkan untuk CatBoost Categorical
features = [c for c in train_full.columns if c not in drop_cols and train_full[c].dtype != 'object' or c == 'nama_pos']

print(f"Total Fitur Final: {len(features)}")
""")

# === Cell 6: STATISTICAL K-MEANS CLUSTERING ===
cell_clustering = nbf.v4.new_code_cell("""# === TAHAP 6: PENGEKSTRAKAN DNA STATISTIK & K-MEANS CLUSTERING (V58) ===
print("Mengekstrak Meta-Features untuk 30 Stasiun...")
meta_features = []
stations = train_full['nama_pos'].unique()

for st in stations:
    tma = train_full[train_full['nama_pos'] == st]['tma_mdpl'].dropna().values
    
    # 1. Volatility
    std_val = np.std(tma)
    # 2. Flashiness (Skewness)
    skew_val = pd.Series(tma).skew()
    
    # Autocorrelation
    acf_vals = acf(tma, nlags=40)
    ac_8 = acf_vals[8] if len(acf_vals) > 8 else 0   # 24 Hours Memory (Lag 8 * 3hr interval in test set)
    ac_40 = acf_vals[40] if len(acf_vals) > 40 else 0 # 120 Hours Memory (Lag 40)
    
    meta_features.append({
        'nama_pos': st,
        'std': std_val,
        'skewness': skew_val,
        'acf_24h': ac_8,
        'acf_120h': ac_40
    })

meta_df = pd.DataFrame(meta_features).set_index('nama_pos')
meta_df.fillna(0, inplace=True)

print("Standarisasi & K-Means Clustering...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(meta_df)

# Kita bagi menjadi 4 Klaster Rezim Alam
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
meta_df['cluster'] = kmeans.fit_predict(X_scaled)

# Buat kamus (dictionary) pemetaan stasiun ke klaster
station_to_cluster = meta_df['cluster'].to_dict()
cluster_to_stations = {i: [] for i in range(4)}
for st, c in station_to_cluster.items():
    cluster_to_stations[c].append(st)

print("\\n=== HASIL PEMBAGIAN KLASTER FISIKA SUNGAI ===")
for c in range(4):
    print(f"\\n💧 KLASTER {c} ({len(cluster_to_stations[c])} Stasiun)")
    print(f"Ciri Fisik Rata-rata: Volatilitas={meta_df[meta_df['cluster']==c]['std'].mean():.2f}, Skewness={meta_df[meta_df['cluster']==c]['skewness'].mean():.2f}")
    print("Anggota:", ", ".join(cluster_to_stations[c][:5]) + ("..." if len(cluster_to_stations[c]) > 5 else ""))
""")

# === Cell 7: Federated CatBoost ===
cell_federated = nbf.v4.new_code_cell("""# === TAHAP 7: FEDERATED CATBOOST TRAINING (4 MODELS) ===
optuna.logging.set_verbosity(optuna.logging.ERROR)
N_TRIALS = 15
cluster_models = {}

print("🚀 Memulai Pelatihan 4 Model CatBoost Raksasa (Federated Training)...\\n")

for c_idx in range(4):
    st_list = cluster_to_stations[c_idx]
    if len(st_list) == 0: continue
        
    print(f"--- MELATIH MODEL KLASTER {c_idx} ({len(st_list)} Stasiun) ---")
    
    # Ambil SEMUA data historis gabungan dari stasiun di klaster ini
    X_cluster = train_full[train_full['nama_pos'].isin(st_list)][features]
    y_cluster = train_full[train_full['nama_pos'].isin(st_list)]['tma_mdpl']
    
    # Urutkan berdasarkan waktu agar split anti-leakage nya valid
    # (Train dari awal sampai 70% masa hidup stasiun tersebut)
    
    # Untuk federated model time-series, kita split berdasarkan proporsi indeks waktu
    idx_tune = int(len(X_cluster) * 0.70)
    idx_val  = int(len(X_cluster) * 0.85)
    
    X_tr = X_cluster.iloc[:idx_tune]
    y_tr = np.clip(y_cluster.iloc[:idx_tune], 0.0001, None)
    
    X_tu = X_cluster.iloc[idx_tune:idx_val]
    y_tu = np.clip(y_cluster.iloc[idx_tune:idx_val], 0.0001, None)
    
    X_va = X_cluster.iloc[idx_val:]
    y_va = np.clip(y_cluster.iloc[idx_val:], 0.0001, None)
    
    # Categorical features index (CatBoost butuh index kolom kategorik)
    cat_features_idx = [X_cluster.columns.get_loc('nama_pos')]
    
    def objective(trial):
        params = {
            'loss_function': 'RMSE',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'depth': trial.suggest_int('depth', 5, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'iterations': 800,
            'early_stopping_rounds': 30,
            'verbose': False,
            'random_seed': 42
        }
        
        train_pool = Pool(X_tr, y_tr, cat_features=cat_features_idx)
        tune_pool = Pool(X_tu, y_tu, cat_features=cat_features_idx)
        
        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=tune_pool)
        
        preds = model.predict(tune_pool)
        return np.sqrt(mean_squared_error(y_tu, preds))

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS)
    
    # 3-in-1 Ensemble CatBoost
    top_trials = sorted(study.trials, key=lambda t: t.value)[:3]
    models = []
    
    X_tr_full = pd.concat([X_tr, X_tu])
    y_tr_full = pd.concat([y_tr, y_tu])
    
    train_full_pool = Pool(X_tr_full, y_tr_full, cat_features=cat_features_idx)
    val_pool = Pool(X_va, y_va, cat_features=cat_features_idx)
        
    for i, trial in enumerate(top_trials):
        params = trial.params.copy()
        params.update({'loss_function': 'RMSE', 'iterations': 1200, 'early_stopping_rounds': 50, 'verbose': False, 'random_seed': 42+i})
        
        model = CatBoostRegressor(**params)
        model.fit(train_full_pool, eval_set=val_pool)
        models.append(model)
        
    # Uji Coba Buta (Holdout)
    ensemble_preds = np.zeros(len(X_va))
    for m in models:
        ensemble_preds += m.predict(X_va) / 3.0
        
    avg_rmse = np.sqrt(mean_squared_error(y_va, ensemble_preds))
    cluster_models[c_idx] = models
    
    print(f"🏅 Klaster {c_idx} TRUE Holdout RMSE: {avg_rmse:.4f}\\n")
""")

# === Cell 8: Prediction & Submission ===
cell_submission = nbf.v4.new_code_cell("""# === TAHAP 8: PREDIKSI MASA DEPAN & SUBMISSION ===
print("Menyuntikkan prediksi gabungan ke Test Set...")
test_preds = np.zeros(len(test_full))

for c_idx in range(4):
    st_list = cluster_to_stations[c_idx]
    if len(st_list) == 0: continue
    
    idx_cluster = test_full['nama_pos'].isin(st_list)
    if idx_cluster.sum() == 0: continue
        
    X_test_cl = test_full.loc[idx_cluster, features]
    
    # 3-in-1 Ensemble Prediction
    cl_preds = np.zeros(len(X_test_cl))
    for m in cluster_models[c_idx]:
        cl_preds += m.predict(X_test_cl) / 3.0
        
    test_full.loc[idx_cluster, 'tma_mdpl'] = cl_preds

sub_pred = test_full[['id', 'tma_mdpl']].copy()
sub = pd.merge(sample_sub[['id']], sub_pred, on='id', how='left')

# Interpolasi darurat jika ada bolong
sub['tma_mdpl'] = sub['tma_mdpl'].fillna(method='ffill').fillna(method='bfill')

output_file = 'submission_V58_Regime_Federation.csv'
sub.to_csv(output_file, index=False)
print(f"✅ BOOM! File '{output_file}' sukses digenerate dan siap mengguncang Kaggle!")
""")


nb.cells = [
    nbf.v4.new_markdown_cell("# 🌊 V58: The Hydrological Regime Federation (Clustered CatBoost)"),
    cell_setup, 
    cell_load, 
    cell_exorcism, 
    cell_baseflow, 
    cell_fusion, 
    cell_clustering,
    cell_federated,
    cell_submission
]

with open('V58_Regime_Federation.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("V58 Notebook generated successfully!")
