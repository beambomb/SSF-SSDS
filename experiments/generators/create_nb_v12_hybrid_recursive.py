import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

def create_v12_notebook():
    nb = new_notebook()
    
    cells = [
        new_markdown_cell('# 🌟 V12 RECURSIVE HYBRID: GWNet Spatial Extractor ➡️ XGBoost\nPipeline paripurna dengan **Recursive Forecasting** untuk menjaga keutuhan fitur fisik (*Autoregressive*). Memanfaatkan Graph WaveNet untuk topologi spasial, dan *loop* prediksi berantai di Inference.'),
        
        new_markdown_cell('## 1. Persiapan Environment & Pemuatan Data'),
        new_code_cell('''!pip install -q geopandas statsmodels
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, Layer, Permute, Reshape
import keras.ops as ops
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

plt.style.use('seaborn-v0_8-whitegrid')

# Auto-Detect Path (Kaggle / Colab / Local)
if not 'path' in locals():
    if os.path.exists('/kaggle/input'):
        dataset_folders = [f for f in os.listdir('/kaggle/input') if os.path.isdir(f'/kaggle/input/{f}')]
        path = f'/kaggle/input/{dataset_folders[0]}/' if dataset_folders else '/kaggle/input/'
    elif os.path.exists('/content'):
        path = '/content/drive/MyDrive/ssf-ssds/'
    else:
        path = './'
print(f"Path dataset aktif: {path}")

# Load Data
train_df = pd.read_csv(path + 'train.csv', parse_dates=['datetime'])
lingkungan_df = pd.read_csv(path + 'data_pendukung/data_lingkungan.csv', parse_dates=['datetime'])
koordinat_df = pd.read_csv(path + 'data_pendukung/koordinat_pos.csv')
print("Data dimuat!")
'''),

        new_markdown_cell('## 2. Target Preprocessing (Outlier Clipping)\nMenggunakan metode IQR Bergulir untuk memangkas *spike* sensor yang rusak tanpa menghilangkan sinyal banjir asli.'),
        new_code_cell('''# Clip TMA (Target) Outliers
for st in train_df['nama_pos'].unique():
    mask = train_df['nama_pos'] == st
    tma_series = train_df.loc[mask, 'tma_mdpl']
    
    q1 = tma_series.rolling(24*7, min_periods=1).quantile(0.25)
    q3 = tma_series.rolling(24*7, min_periods=1).quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 3 * iqr
    
    rolling_med = tma_series.rolling(24, min_periods=1).median()
    train_df.loc[mask, 'tma_mdpl'] = np.where((tma_series > upper_bound) & (tma_series.notnull()), rolling_med, tma_series)

print("Target TMA Outliers telah dibersihkan!")
'''),

        new_markdown_cell('## 3. Massive Feature Engineering (Pure Exogenous)\nMenggabungkan seluruh "Harta Karun" cuaca: Mikro, Makro, Tanah, Siklus Waktu, *Rolling*, dan *Lag* (sebagai memori air hulu-hilir).'),
        new_code_cell('''full_env = lingkungan_df.copy()

# A. Iklim Mikro Instan & Siklus Waktu
full_env['dew_point_spread'] = full_env['temperature_c'] - full_env['dew_point_c']
full_env['hour_sin'] = np.sin(2 * np.pi * full_env['datetime'].dt.hour / 24.0)
full_env['hour_cos'] = np.cos(2 * np.pi * full_env['datetime'].dt.hour / 24.0)
full_env['month_sin'] = np.sin(2 * np.pi * full_env['datetime'].dt.month / 12.0)
full_env['month_cos'] = np.cos(2 * np.pi * full_env['datetime'].dt.month / 12.0)
full_env['wind_sin'] = np.sin(np.radians(full_env['wind_direction_deg']))
full_env['wind_cos'] = np.cos(np.radians(full_env['wind_direction_deg']))

# B. Rolling & Lags (Memori Sungai) - DIHITUNG PER STASIUN
print("Menghitung Rolling & Lags... ini akan memakan waktu sebentar.")
for st, group in full_env.groupby('nama_pos'):
    idx = group.index
    
    # 1. Rolling Akumulasi Hujan (Kelembaban sistem)
    full_env.loc[idx, 'rain_roll_12h'] = group['rainfall_mm'].rolling(12, min_periods=1).sum()
    full_env.loc[idx, 'rain_roll_24h'] = group['rainfall_mm'].rolling(24, min_periods=1).sum()
    full_env.loc[idx, 'rain_roll_72h'] = group['rainfall_mm'].rolling(72, min_periods=1).sum()
    full_env.loc[idx, 'rain_roll_168h'] = group['rainfall_mm'].rolling(168, min_periods=1).sum() # 1 Minggu
    full_env.loc[idx, 'rain_variance_24h'] = group['rainfall_mm'].rolling(24, min_periods=1).var().fillna(0)
    
    # 2. Lags Hujan Eksogen (Krusial untuk rambatan Hulu -> Hilir)
    full_env.loc[idx, 'rain_lag_24h'] = group['rainfall_mm'].shift(24) # Hujan 1 hari lalu
    full_env.loc[idx, 'rain_lag_48h'] = group['rainfall_mm'].shift(48) # Hujan 2 hari lalu
    full_env.loc[idx, 'rain_lag_72h'] = group['rainfall_mm'].shift(72) # Hujan 3 hari lalu
    
    # 3. Profil Tanah (Kecepatan Infiltrasi Air)
    full_env.loc[idx, 'soil_diff_0_7cm_24h'] = group['soil_moisture_0_7cm'].diff(24)

# Gabungkan dengan target (TMA)
df_all = pd.merge(train_df, full_env, on=['datetime', 'nama_pos'], how='inner')

# C. Lags Autoregressive (Memori Fisik Tinggi Air)
df_all['tma_lag_24h'] = df_all.groupby('nama_pos')['tma_mdpl'].shift(8) # Shift 8 step (8 * 3 jam/hari = 24 jam)
df_all['tma_lag_48h'] = df_all.groupby('nama_pos')['tma_mdpl'].shift(16)
df_all['tma_lag_72h'] = df_all.groupby('nama_pos')['tma_mdpl'].shift(24)

# D. One-Hot Encoding (OHE) Stasiun
ohe_pos = pd.get_dummies(df_all['nama_pos'], prefix='pos', drop_first=False)
df_all = pd.concat([df_all, ohe_pos], axis=1)

print(f"Dataset Tabular dengan Lags & OHE siap! Shape: {df_all.shape}")
'''),

        new_markdown_cell('## 4. Persiapan Tensor GWNet\nMengubah format *tabular* menjadi *Tensor 3D/4D* (Urutan Waktu per Stasiun) untuk dilatih oleh Graph WaveNet.'),
        new_code_cell('''# Ekstrak waktu dan stasiun
times = sorted(df_all['datetime'].unique())
stations = sorted(df_all['nama_pos'].unique())
NUM_STATIONS = len(stations)
SEQ_LEN = 24 # Sequence 24 Jam

# Membentuk Pivot Table agar data terurut secara Spatio-Temporal
gwnet_feats = ['tma_mdpl', 'rainfall_mm', 'soil_moisture_0_7cm', 'nino_34']
NUM_FEATS = len(gwnet_feats)

pivot_cols = list(dict.fromkeys(gwnet_feats + ['tma_mdpl']))
pivot_df = df_all.pivot(index='datetime', columns='nama_pos', values=pivot_cols)
pivot_df.fillna(method='ffill', inplace=True)
pivot_df.fillna(method='bfill', inplace=True)

# Membangun Tensor
X_list, y_list, time_idx = [], [], []

# Geser jendela (Sliding Window) sebesar 24 jam
print("Membentuk Tensor Sequences... (Membutuhkan sedikit RAM)")
for i in range(SEQ_LEN, len(times)):
    # Data X: (SEQ_LEN, NUM_STATIONS, NUM_FEATS)
    x_window = []
    for f in gwnet_feats:
        feat_data = pivot_df[f].iloc[i-SEQ_LEN:i].values # (SEQ_LEN, NUM_STATIONS)
        x_window.append(feat_data)
    
    # Stack features menjadi tensor 3D: (SEQ_LEN, NUM_STATIONS, NUM_FEATS)
    X_list.append(np.stack(x_window, axis=-1))
    
    # Target Y: TMA pada waktu t (i) untuk seluruh stasiun
    y_list.append(pivot_df['tma_mdpl'].iloc[i].values) # (NUM_STATIONS,)
    time_idx.append(times[i])

X_tensor = np.array(X_list, dtype=np.float32) # Shape: (Batches, 24, 30, 5)
y_tensor = np.array(y_list, dtype=np.float32) # Shape: (Batches, 30)

print(f"Tensor X GWNet: {X_tensor.shape}")
print(f"Tensor Y GWNet: {y_tensor.shape}")
'''),

        new_markdown_cell('## 5. Arsitektur & Training GWNet (Feature Extractor)\nMelatih GWNet secara singkat untuk menangkap struktur spasial hulu-hilir, lalu memotong *layer*-nya untuk menghasilkan **Embeddings 32-D**.'),
        new_code_cell('''# Matriks Adjacency Dummy (Adaptive Graph akan mempelajari struktur aslinya secara otomatis)
A_norm = np.ones((NUM_STATIONS, NUM_STATIONS), dtype=np.float32) / NUM_STATIONS

class TemporalConvNet(Layer):
    def __init__(self, filters, kernel_size=2, dilation_rate=1, **kwargs):
        super(TemporalConvNet, self).__init__(**kwargs)
        self.conv = Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='causal', activation='relu')
    def call(self, inputs):
        n, t, f = NUM_STATIONS, SEQ_LEN, inputs.shape[-1]
        x = ops.reshape(inputs, [-1, t, f])
        x = self.conv(x)
        return ops.reshape(x, [-1, n, t, self.conv.filters])

class AdaptiveGraphConv(Layer):
    def __init__(self, A_matrix, embed_dim=10, **kwargs):
        super(AdaptiveGraphConv, self).__init__(**kwargs)
        self.A = ops.convert_to_tensor(A_matrix, dtype='float32')
        self.embed_dim = embed_dim
    def build(self, input_shape):
        self.E1 = self.add_weight(shape=(NUM_STATIONS, self.embed_dim), initializer='random_normal')
        self.E2 = self.add_weight(shape=(NUM_STATIONS, self.embed_dim), initializer='random_normal')
    def call(self, inputs):
        A_adapt = tf.nn.softmax(tf.nn.relu(ops.matmul(self.E1, ops.transpose(self.E2))), axis=1)
        return ops.einsum('vw,bvtf->bwtf', self.A + A_adapt, inputs)

# Struktur GWNet
inputs = Input(shape=(SEQ_LEN, NUM_STATIONS, NUM_FEATS))
x = Permute((2, 1, 3))(inputs) # Ubah ke (Batch, Nodes, Waktu, Fitur)
x = TemporalConvNet(16, 2, 1)(x)
x = AdaptiveGraphConv(A_norm, 8)(x)
x = TemporalConvNet(32, 2, 2)(x)
x = AdaptiveGraphConv(A_norm, 8)(x)

# EKSTRAKSI EMBEDDING KRUSIAL
# Kita ambil hasil konvolusi pada time-step terakhir sebagai representasi absolut dari sistem sungai
spatial_embeddings = tf.keras.layers.Lambda(lambda t: t[:, :, -1, :], name='spatial_embeddings')(x) # (Batch, Nodes, 32)

outputs = Dense(1)(spatial_embeddings)
outputs = Reshape((NUM_STATIONS,))(outputs)

gwnet_model = Model(inputs, outputs)
gwnet_model.compile(optimizer='adam', loss='mse')

# Training Cepat GWNet (Cukup 15 Epochs agar model mengerti spasial dasar)
print("Melatih GWNet Feature Extractor (15 Epochs)...")
history = gwnet_model.fit(
    X_tensor, y_tensor, 
    batch_size=128, 
    epochs=15, 
    validation_split=0.2, 
    verbose=1
)

# Ambil model potongannya
extractor = Model(inputs=gwnet_model.input, outputs=gwnet_model.get_layer('spatial_embeddings').output)
print("Extractor siap!")
'''),

        new_markdown_cell('## 6. Produksi Spatial Embeddings & Fusion Tabular\nMenembakkan data ke *Extractor* untuk mendapatkan nilai *Embeddings* riil (bukan *dummy* lagi!), lalu menggabungkannya kembali ke format Tabular XGBoost.'),
        new_code_cell('''print("Menghasilkan Spatial Embeddings (32 Dimensi) untuk seluruh waktu...")
# Prediksi Embeddings: Hasilnya (Batches, 30 Stasiun, 32 Dimensi)
raw_embeddings = extractor.predict(X_tensor, batch_size=256)

# Kita perlu meratakannya (Flatten) kembali ke format df_all (Baris = Waktu x Stasiun)
# Karena X_tensor diurutkan berdasarkan time_idx lalu stations, kita bangun DataFrame perantara
emb_rows = []
for i, t in enumerate(time_idx):
    for j, st in enumerate(stations):
        emb_values = raw_embeddings[i, j, :]
        row_dict = {'datetime': t, 'nama_pos': st}
        for dim in range(32):
            row_dict[f'gwnet_emb_{dim}'] = emb_values[dim]
        emb_rows.append(row_dict)

df_emb = pd.DataFrame(emb_rows)

# Gabungkan dengan Tabular DataFrame
# Gunakan inner join untuk membuang SEQ_LEN hari pertama yang tidak punya sequence
df_final = pd.merge(df_all, df_emb, on=['datetime', 'nama_pos'], how='inner')

print(f"Fusion Berhasil! Dataset Hibrida Final: {df_final.shape}")
'''),

        new_markdown_cell('## 7. XGBoost Training (True Hybrid)\nMelatih rajanya *Tabular Regression* menggunakan seluruh porsi makanan yang telah disiapkan. (Hujan Lag, Cuaca, Tanah, OHE, dan 32 Dimensi GWNet).'),
        new_code_cell('''print("Menyiapkan Porsi XGBoost (~95 Fitur)...")
# A. Daftarkan Semua Fitur Cuaca, Tanah, Lag & Rolling
xgb_features = [
    'tma_lag_24h', 'tma_lag_48h', 'tma_lag_72h',
    'rainfall_mm', 'humidity_pct', 'cloud_cover_pct', 'solar_radiation_mj_m2',
    'soil_moisture_0_7cm', 'soil_moisture_7_28cm', 'soil_moisture_28_100cm', 'soil_moisture_100_255cm',
    'nino_34', 'rmm1', 'rmm2', 'mjo_amplitude', 'mjo_active',
    'month_sin', 'month_cos',
    'rain_roll_12h', 'rain_roll_24h', 'rain_roll_72h', 'rain_roll_168h', 'rain_variance_24h',
    'rain_lag_24h', 'rain_lag_48h', 'rain_lag_72h', 'soil_diff_0_7cm_24h'
]

# B. Daftarkan 30 Fitur One-Hot Encoding Stasiun
pos_cols = [c for c in df_final.columns if c.startswith('pos_')]
xgb_features.extend(pos_cols)

# C. Daftarkan 32 Fitur GWNet Embeddings
gwnet_cols = [f'gwnet_emb_{i}' for i in range(32)]
xgb_features.extend(gwnet_cols)

print(f"Total Fitur yang dilempar ke XGBoost: {len(xgb_features)}")

# D. Pembersihan Baris (Missing Values dari Lags)
df_clean = df_final.dropna(subset=['tma_mdpl'] + xgb_features)

# =========================================================
# 🚨 KRUSIAL: Pengurutan Waktu agar K-Fold tidak bocor stasiun!
df_clean = df_clean.sort_values('datetime').reset_index(drop=True)
# =========================================================

X = df_clean[xgb_features].astype(float)
y = df_clean['tma_mdpl'] # KEMBALI KE ABSOLUT TARGET (MDPL)

print(f"Data Bersih Siap Latih: X={X.shape}, y={y.shape}")

# E. K-Fold Training Loop (Pembelahan Waktu Murni)
kf = KFold(n_splits=3, shuffle=False)
models_xgb = []
rmse_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\\n--- Fold {fold + 1} ---")
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    # Gunakan parameter yang agresif karena fitur sangat kaya
    model = xgb.XGBRegressor(
        n_estimators=1000, 
        learning_rate=0.02,
        max_depth=8, 
        subsample=0.8,
        colsample_bytree=0.7,
        objective='reg:squarederror',
        eval_metric='rmse',
        early_stopping_rounds=50,
        random_state=42,
        tree_method='hist' 
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=200
    )
    
    val_rmse = root_mean_squared_error(y_va, model.predict(X_va))
    rmse_scores.append(val_rmse)
    models_xgb.append(model)

print(f"\\n🏆 Rata-rata Validation RMSE (Ultimate V11): {np.mean(rmse_scores):.4f}")
'''),

        new_markdown_cell('## 8. Evaluasi & Visualisasi Bukti Hibrida\nApakah XGBoost memakan *Embeddings* dari GWNet? Mari kita buktikan di *Feature Importance*!'),
        new_code_cell('''fig, ax = plt.subplots(1, 2, figsize=(20, 7))

# A. Feature Importance
xgb.plot_importance(models_xgb[-1], max_num_features=20, height=0.6, ax=ax[0], color='crimson')
ax[0].set_title('Top 20 Feature Importance (Perhatikan Lags & GWNet!)')

# B. Actual vs Predicted (Sampel Babat - Fold Validasi)
# Karena OHE dipakai, kita filter baris spesifik Babat
st_mask = df_clean['pos_Babat'] == 1.0 
X_babat = df_clean.loc[st_mask, xgb_features].astype(float)
y_babat = df_clean.loc[st_mask, 'tma_mdpl']

if len(X_babat) > 300:
    # Ambil 300 jam terakhir (Mewakili bagian dari Validation Set di akhir waktu)
    X_sub = X_babat.iloc[-300:]
    y_sub = y_babat.iloc[-300:].values
    
    # Model menebak absolut MDPL, langsung plot
    pred_sub = models_xgb[-1].predict(X_sub)
    
    ax[1].plot(y_sub, label='Actual TMA (Babat)', color='navy', alpha=0.8, linewidth=2)
    ax[1].plot(pred_sub, label='Hybrid XGBoost Predict', color='orange', alpha=0.9, linestyle='--', linewidth=2.5)
    ax[1].set_title('Actual vs Predicted Hibrida (Akhir Validation Set - Babat)')
    ax[1].legend()

plt.tight_layout()
plt.show()
'''),

        new_markdown_cell('## 9. Recursive Inference & Submission\nInilah keajaiban V12! Kita menggunakan **Recursive Forecasting**. Model menebak hari H, menyimpan tebakannya, lalu menggunakannya sebagai `tma_lag` untuk menebak H+1. GWNet juga akan membaca hasil tebakan ini secara *real-time*.'),
        new_code_cell('''import time
print("Memulai proses Recursive Inference Test Data...")
start_time = time.time()

# 1. Muat Data Test
test_df = pd.read_csv(path + 'test.csv')
test_df['datetime'] = pd.to_datetime(test_df['id'].str.split(' - ', n=1).str[0])
test_df['nama_pos'] = test_df['id'].str.split(' - ', n=1).str[1]
print(f"Data Test dimuat: {test_df.shape}")

# 2. Gabungkan Train & Test
# Target tma_mdpl di test set masih NaN, kita akan mengisinya pelan-pelan
gabungan_df = pd.concat([train_df[['datetime', 'nama_pos', 'tma_mdpl']], test_df[['datetime', 'nama_pos']]], ignore_index=True)
gabungan_df.sort_values(['nama_pos', 'datetime'], inplace=True)
gabungan_df = pd.merge(gabungan_df, full_env, on=['datetime', 'nama_pos'], how='left')

# 3. OHE Stasiun
ohe_pos_test = pd.get_dummies(gabungan_df['nama_pos'], prefix='pos', drop_first=False)
gabungan_df = pd.concat([gabungan_df, ohe_pos_test], axis=1)

# 4. Siapkan Matriks Pivot untuk Akses Cepat (O(1) lookup)
all_times = sorted(gabungan_df['datetime'].unique())
test_times = sorted(test_df['datetime'].unique())

pivot_feats = {}
for f in gwnet_feats:
    if f != 'tma_mdpl':
        p = gabungan_df.pivot(index='datetime', columns='nama_pos', values=f)
        p.fillna(method='ffill', inplace=True)
        p.fillna(method='bfill', inplace=True)
        pivot_feats[f] = p

# Matriks target ini yang akan terus di-update secara live!
pivot_target = gabungan_df.pivot(index='datetime', columns='nama_pos', values='tma_mdpl')
pivot_target.fillna(method='ffill', inplace=True) # Isi awal jika ada bolong

# 5. LOOP RECURSIVE FORECASTING
print(f"Looping sebanyak {len(test_times)} timesteps...")
for step, t in enumerate(test_times):
    if step % 100 == 0:
        print(f"Memproses Step {step}/{len(test_times)} - Waktu: {t}")
        
    t_idx = all_times.index(t)
    mask_t = gabungan_df['datetime'] == t
    
    # A. Update Lags di gabungan_df untuk waktu t
    for lag_name, hours in zip(['tma_lag_24h', 'tma_lag_48h', 'tma_lag_72h'], [24, 48, 72]):
        past_t = t - pd.Timedelta(hours=hours)
        if past_t in pivot_target.index:
            gabungan_df.loc[mask_t, lag_name] = gabungan_df.loc[mask_t, 'nama_pos'].map(pivot_target.loc[past_t])
            
    # B. Ekstrak Tensor Spasial (SEQ_LEN=24) untuk waktu t
    x_win = []
    for f in gwnet_feats:
        if f == 'tma_mdpl':
            # Gunakan pivot_target yang terus diupdate
            feat_data = pivot_target.iloc[t_idx - SEQ_LEN + 1 : t_idx + 1].values
        else:
            feat_data = pivot_feats[f].iloc[t_idx - SEQ_LEN + 1 : t_idx + 1].values
        x_win.append(feat_data)
        
    x_tensor = np.stack(x_win, axis=-1)
    x_tensor = np.expand_dims(x_tensor, axis=0) # (1, 24, 30, 4)
    
    # C. Ekstrak Embeddings
    raw_emb = extractor.predict(x_tensor, verbose=0)
    
    # D. Masukkan Embeddings ke DataFrame Baris T
    row_feats = gabungan_df[mask_t].copy()
    for j, st in enumerate(stations):
        mask_st = row_feats['nama_pos'] == st
        for dim in range(32):
            row_feats.loc[mask_st, f'gwnet_emb_{dim}'] = raw_emb[0, j, dim]
            
    # E. Prediksi XGBoost!
    X_test = row_feats[xgb_features].astype(float)
    y_pred = models_xgb[-1].predict(X_test)
    
    # F. SIMPAN KEMBALI HASIL PREDIKSI (INILAH RECURSIVE!)
    # Simpan di gabungan_df
    gabungan_df.loc[mask_t, 'tma_mdpl'] = y_pred
    
    # Simpan di pivot_target agar bisa dipanggil GWNet di step berikutnya
    for j, st in enumerate(stations):
        # Karena kita iterate enumerate stations, j sesuai abjad, y_pred urutannya sesuai X_test yang diambil dari gabungan_df
        # Tunggu, urutan X_test belum tentu sama dengan urutan 'stations'
        # Agar 100% aman:
        pass
    
    # Safe assign
    for st_val, y_val in zip(row_feats['nama_pos'], y_pred):
        pivot_target.at[t, st_val] = y_val

print(f"Recursive Inference Selesai! Waktu total: {time.time() - start_time:.1f} detik.")

# 6. Format Submission
test_final = gabungan_df[gabungan_df['datetime'].isin(test_times)].copy()
submission = pd.merge(test_df[['id', 'datetime', 'nama_pos']], test_final[['datetime', 'nama_pos', 'tma_mdpl']], on=['datetime', 'nama_pos'], how='left')
submission = submission[['id', 'tma_mdpl']]

submission.to_csv('submission_v12.csv', index=False)
print("SELESAI! File submission_v12.csv telah berhasil dibuat!")
'''),
        new_markdown_cell('## 10. Visualisasi Hasil Recursive Forecasting\nMari kita lihat secara visual apakah tebakan berantai kita di data *Test* membentuk kurva sungai yang wajar atau malah melenceng (*error accumulation*).'),
        new_code_cell('''import matplotlib.pyplot as plt

# Pilih 3 stasiun sampel yang sering banjir
sample_stations = ['Babat', 'Bojonegoro - Kali Kethek', 'Jurug']

plt.figure(figsize=(20, 12))
for i, st in enumerate(sample_stations):
    plt.subplot(3, 1, i+1)
    
    # Filter data test untuk stasiun ini
    st_data = test_final[test_final['nama_pos'] == st].sort_values('datetime')
    
    plt.plot(st_data['datetime'], st_data['tma_mdpl'], label=f'Predicted TMA (MDPL) - {st}', color='crimson', linewidth=2)
    plt.title(f'Hasil Prediksi Recursive (Test Set): {st}', fontsize=14, fontweight='bold')
    plt.ylabel('TMA (MDPL)')
    plt.grid(True, alpha=0.3)
    plt.legend()

plt.tight_layout()
plt.show()
''')
    ]
    
    nb.cells.extend(cells)
    with open('V12_Hybrid_GWNet_XGBoost_Recursive.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == '__main__':
    create_v12_notebook()
    print("Notebook V12_Hybrid_GWNet_XGBoost_Recursive.ipynb berhasil dibuat!")
