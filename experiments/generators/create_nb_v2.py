import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()

nb.cells.extend([
    new_markdown_cell('# 🌊 Advanced TMA Forecasting: Spatial-Temporal & Recursive Inference\n\nNotebook ini adalah versi mutakhir (V2) yang mencakup:\n1. **Long-term Antecedent Precipitation Index (API):** Akumulasi cuaca 3 & 7 hari terakhir.\n2. **Climate Anomalies:** Efek El Nino/La Nina & MJO selama 30 hari terakhir.\n3. **Autoregressive Target (Target Lags):** Menggunakan histori TMA kemarin.\n4. **Spatial Cross-Lags (Hulu-Hilir):** Memanfaatkan rambatan air dari stasiun dataran tinggi ke stasiun lainnya.\n5. **Recursive Inference Loop:** Mesin prediksi beruntun untuk menghindari kebocoran data (Data Leakage) saat memprediksi masa depan.'),
    
    new_code_cell('''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import LabelEncoder
import warnings
import gc

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 150)'''),
    
    new_markdown_cell('## 1. Setup & Data Loading'),
    new_code_cell('''# Mount Google Drive (hanya jalan di Colab)
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("Google Drive mounted successfully!")
except:
    print("Not running in Google Colab, skipping drive mount.")

# Path ke folder ssf-ssds di root Google Drive Anda
path = '/content/drive/MyDrive/ssf-ssds/'
# Jika dijalankan di PC lokal, ganti path ke direktori lokal (misal: path = './')

# Load data utama
print("Loading data...")
train_df = pd.read_csv(path + 'train.csv', parse_dates=['datetime'])
test_df = pd.read_csv(path + 'test.csv')
sample_sub = pd.read_csv(path + 'sample_submission.csv')

lingkungan_df = pd.read_csv(path + 'data_pendukung/data_lingkungan.csv', parse_dates=['datetime'])
koordinat_df = pd.read_csv(path + 'data_pendukung/koordinat_pos.csv')

print(f'Train shape: {train_df.shape}')
print(f'Lingkungan shape: {lingkungan_df.shape}')'''),

    new_markdown_cell('## 2. Advanced Feature Engineering (Environment & Climate)'),
    new_code_cell('''# Kita proses data lingkungan terlebih dahulu sebelum digabung dengan target.
lingkungan_df = lingkungan_df.sort_values(['nama_pos', 'datetime']).reset_index(drop=True)

# Definisikan kolom-kolom penting
weather_cols = ['rainfall_mm', 'humidity_pct', 'temperature_c', 'wind_speed_kmh', 
                'soil_moisture_0_7cm', 'surface_pressure_hpa', 'solar_radiation_mj_m2']

# Cek apakah kolom iklim ada di data lingkungan. Jika ada, masukkan.
climate_cols = [c for c in lingkungan_df.columns if c in ['nino_34', 'mjo_phase', 'rmm1', 'rmm2']]

def build_advanced_env_features(env_df):
    agg_df = env_df[['datetime', 'nama_pos']].copy()
    grouped = env_df.groupby('nama_pos')
    
    # 1. Short-term (6 hours) - Sama seperti V1
    for col in weather_cols:
        agg_df[f'{col}_mean_6h'] = grouped[col].transform(lambda x: x.rolling(6, min_periods=1).mean())
        agg_df[f'{col}_sum_6h'] = grouped[col].transform(lambda x: x.rolling(6, min_periods=1).sum())
        
    # 2. Long-term API (Antecedent Precipitation Index)
    # Akumulasi 3 hari (72 jam) dan 7 hari (168 jam) untuk hujan
    if 'rainfall_mm' in env_df.columns:
        agg_df['rainfall_sum_72h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(72, min_periods=1).sum())
        agg_df['rainfall_sum_168h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(168, min_periods=1).sum())
        
    # Kejenuhan tanah (soil moisture) 3 hari terakhir
    if 'soil_moisture_0_7cm' in env_df.columns:
        agg_df['soil_moisture_mean_72h'] = grouped['soil_moisture_0_7cm'].transform(lambda x: x.rolling(72, min_periods=1).mean())
        
    # 3. Climate Anomalies (30 days = 720 hours)
    for c_col in climate_cols:
        agg_df[f'{c_col}_mean_720h'] = grouped[c_col].transform(lambda x: x.rolling(720, min_periods=1).mean())
        
    return agg_df

print("Membangun fitur lingkungan jangka menengah/panjang (butuh beberapa saat)...")
env_feat = build_advanced_env_features(lingkungan_df)

# Filter HANYA pada jam pencatatan TMA (06:00, 12:00, 18:00) untuk menghemat RAM
env_feat = env_feat[env_feat['datetime'].dt.hour.isin([6, 12, 18])].reset_index(drop=True)
print("Fitur Lingkungan Selesai!")
'''),

    new_markdown_cell('## 3. Ekstrak Target Lags & Spatial Hulu-Hilir di Data Latih'),
    new_code_cell('''# Identifikasi Hulu vs Hilir berdasarkan rata-rata TMA tertinggi di Training Data
mean_elevations = train_df.groupby('nama_pos')['tma_mdpl'].mean().sort_values(ascending=False)
hulu_stations = mean_elevations.index[:5].tolist() # 5 stasiun tertinggi dianggap hulu
print("Top 5 Stasiun Hulu (Proxy Elevasi Tertinggi):", hulu_stations)

# Sort training data
train_df = train_df.sort_values(['nama_pos', 'datetime']).reset_index(drop=True)

# Membuat target lags murni (berdasarkan urutan observasi, bukan jam pasti karena ada jeda malam hari)
grouped_train = train_df.groupby('nama_pos')
train_df['tma_lag1'] = grouped_train['tma_mdpl'].shift(1)
train_df['tma_lag2'] = grouped_train['tma_mdpl'].shift(2)
train_df['tma_lag3'] = grouped_train['tma_mdpl'].shift(3)

# Hitung Rata-rata TMA Hulu pada saat (t-1)
# Ini adalah FITUR SPATIAL: merambatkan informasi air turun dari gunung ke stasiun lainnya
tma_hulu_df = train_df[train_df['nama_pos'].isin(hulu_stations)][['datetime', 'tma_lag1']]
hulu_lag1_mean = tma_hulu_df.groupby('datetime')['tma_lag1'].mean().reset_index()
hulu_lag1_mean.rename(columns={'tma_lag1': 'mean_hulu_tma_lag1'}, inplace=True)

# Merge fitur spatial ke train_df
train_df = pd.merge(train_df, hulu_lag1_mean, on='datetime', how='left')

# Jika lag kosong (baris-baris pertama), fill dengan backfill
train_df['tma_lag1'] = train_df.groupby('nama_pos')['tma_lag1'].bfill()
train_df['tma_lag2'] = train_df.groupby('nama_pos')['tma_lag2'].bfill()
train_df['tma_lag3'] = train_df.groupby('nama_pos')['tma_lag3'].bfill()
train_df['mean_hulu_tma_lag1'] = train_df['mean_hulu_tma_lag1'].fillna(method='bfill')

# Gabungkan dengan fitur lingkungan
train_master = pd.merge(train_df, env_feat, on=['datetime', 'nama_pos'], how='left')
train_master = pd.merge(train_master, koordinat_df, on='nama_pos', how='left')

# Hapus missing values awal akibat proses rolling/bfill yang gagal
train_master = train_master.dropna(subset=['tma_mdpl', 'tma_lag1', 'rainfall_mm_mean_6h']).reset_index(drop=True)
'''),

    new_markdown_cell('## 4. Time Encoding & Label Encoding'),
    new_code_cell('''# Buat fitur waktu yang siklikal
def add_time_features(df):
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12.0)
    return df

train_master = add_time_features(train_master)

# Encode Categorical
le = LabelEncoder()
# Kita fit pada semua station yang unik
all_stations = pd.concat([train_df['nama_pos'], test_df['id'].str.split(' - ', expand=True)[1]]).unique()
le.fit(all_stations)

train_master['nama_pos_encoded'] = le.transform(train_master['nama_pos'])
'''),

    new_markdown_cell('## 5. Model Training (XGBoost)'),
    new_code_cell('''# Siapkan X dan y
drop_cols = ['datetime', 'nama_pos', 'tma_mdpl', 'id']
features = [c for c in train_master.columns if c not in drop_cols]

X = train_master[features]
y = train_master['tma_mdpl']

print(f"Banyaknya fitur: {len(features)}")
print(features)

# Setup K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
models = []
scores = []

# Parameter dituning untuk lambat & stabil (Mencegah Overfitting)
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'learning_rate': 0.015,
    'max_depth': 6,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.5,
    'reg_lambda': 1.0,
    'tree_method': 'hist',
    'random_state': 42
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"--- Fold {fold+1} ---")
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_va, label=y_va)
    
    model = xgb.train(
        params, 
        dtrain, 
        num_boost_round=10000, 
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=200,
        verbose_eval=200
    )
    
    val_preds = model.predict(dval)
    fold_rmse = root_mean_squared_error(y_va, val_preds)
    scores.append(fold_rmse)
    models.append(model)
    print(f"Fold {fold+1} RMSE: {fold_rmse:.4f}\\n")

print(f"Average CV RMSE: {np.mean(scores):.4f}")

plt.figure(figsize=(10, 8))
xgb.plot_importance(models[0], max_num_features=25, height=0.5)
plt.title('Top 25 Feature Importance (V2)')
plt.show()
'''),

    new_markdown_cell('## 6. Recursive Inference untuk Data Test (The Masterpiece) 👑\n\nUntuk menghindari error meroket karena *data leakage*, kita akan memprediksi data test selangkah demi selangkah sesuai kronologi waktu. Prediksi jam 06:00 akan dimasukkan ke dalam histori untuk digunakan sebagai fitur (lag) pada prediksi jam 12:00, dan seterusnya secara berulang (*recursive*).'),
    new_code_cell('''# 1. Parse Test Data
test_parsed = test_df.copy()
split_info = test_parsed['id'].str.split(' - ', n=1, expand=True)
test_parsed['datetime'] = pd.to_datetime(split_info[0])
test_parsed['nama_pos'] = split_info[1]
test_parsed['nama_pos_encoded'] = le.transform(test_parsed['nama_pos'])
test_parsed = add_time_features(test_parsed)

# Gabungkan dengan koordinat & fitur lingkungan statis dari `env_feat`
test_parsed = pd.merge(test_parsed, koordinat_df, on='nama_pos', how='left')
test_parsed = pd.merge(test_parsed, env_feat, on=['datetime', 'nama_pos'], how='left')

# 2. Siapkan Buffer Historis TMA dari data train (Data Asli Terakhir)
last_train_records = train_df.sort_values('datetime').groupby('nama_pos').tail(3)
tma_history = {}
for station in all_stations:
    station_data = last_train_records[last_train_records['nama_pos'] == station]['tma_mdpl'].values
    if len(station_data) == 3:
        tma_history[station] = list(station_data) # [lag3, lag2, lag1]
    else:
        # Jika stasiun tidak ada di train (jarang terjadi), isi dengan median global
        median_val = train_df['tma_mdpl'].median()
        tma_history[station] = [median_val, median_val, median_val]

# 3. Eksekusi Recursive Loop
unique_dts = sorted(test_parsed['datetime'].unique())
test_predictions = []
print(f"Memulai loop inference untuk {len(unique_dts)} langkah waktu unik...")

for i, current_dt in enumerate(unique_dts):
    # Ambil baris-baris test untuk waktu ini
    current_batch = test_parsed[test_parsed['datetime'] == current_dt].copy()
    
    # Hitung rata-rata TMA Hulu berdasarkan histori terkini (lag1)
    hulu_current_lags = [tma_history[s][-1] for s in hulu_stations if s in tma_history]
    mean_hulu = np.mean(hulu_current_lags) if len(hulu_current_lags) > 0 else 0
    
    # Inject Autoregressive Features ke current_batch
    tma_lag1_list = []
    tma_lag2_list = []
    tma_lag3_list = []
    
    for station in current_batch['nama_pos']:
        hist = tma_history[station]
        tma_lag1_list.append(hist[-1])
        tma_lag2_list.append(hist[-2])
        tma_lag3_list.append(hist[-3])
        
    current_batch['tma_lag1'] = tma_lag1_list
    current_batch['tma_lag2'] = tma_lag2_list
    current_batch['tma_lag3'] = tma_lag3_list
    current_batch['mean_hulu_tma_lag1'] = mean_hulu
    
    # Pastikan urutan fitur sama persis dengan saat training
    X_current = current_batch[features]
    
    # Prediksi menggunakan rata-rata ensemble dari 5 Fold
    preds = np.zeros(len(X_current))
    dmatrix_current = xgb.DMatrix(X_current)
    for model in models:
        preds += model.predict(dmatrix_current) / len(models)
        
    current_batch['tma_mdpl_pred'] = preds
    
    # Simpan hasil untuk submission
    for _, row in current_batch.iterrows():
        test_predictions.append({
            'id': row['id'],
            'tma_mdpl': row['tma_mdpl_pred']
        })
        
        # Update Buffer Histori (Buang yang paling lama, masukkan yang baru)
        st = row['nama_pos']
        tma_history[st].pop(0) # buang lag3
        tma_history[st].append(row['tma_mdpl_pred']) # jadikan ini lag1 berikutnya
        
    if (i+1) % 50 == 0:
        print(f"Selesai langkah {i+1} dari {len(unique_dts)}...")

print("Recursive Inference Selesai!")
'''),

    new_markdown_cell('## 7. Submission Preparation'),
    new_code_cell('''# Buat dataframe dari dictionary hasil loop
sub_pred = pd.DataFrame(test_predictions)

# Pasangkan urutannya agar 100% sama persis dengan `sample_submission.csv`
sub = pd.merge(sample_sub[['id']], sub_pred, on='id', how='left')

# Jika ada NaN karena sesuatu (misal datetime bolong), isi dengan iterpolasi/bfill
if sub['tma_mdpl'].isna().sum() > 0:
    sub['tma_mdpl'] = sub['tma_mdpl'].fillna(method='bfill')

print("Sample Submission:")
display(sub.head(10))

# Ekspor
sub.to_csv('submission_xgboost_advanced.csv', index=False)
print("\\n✅ File 'submission_xgboost_advanced.csv' (V2) siap disubmit ke Kaggle!")
''')
])

with open('XGBoost_Advanced_TMA.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
