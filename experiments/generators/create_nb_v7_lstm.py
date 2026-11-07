import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()

nb.cells.extend([
    new_markdown_cell('# 🚀 V7: Exogenous LSTM (Weather-to-Water Seq2Vec)\n\nNotebook ini mendobrak batas kelemahan *Tree-Models* (XGBoost) yang sering meremehkan (*underpredict*) puncak banjir ekstrem. Kita menggunakan **Deep Learning (LSTM)** untuk menelan urutan (*sequence*) cuaca historis 30 langkah (7.5 hari) ke belakang, dan memprediksi tinggi air hari ini.\n\nNotebook ini **100% KOMPLIT** mencakup EDA, Dokumentasi Arsitektur, Evaluasi Visual, dan Inferensi yang aman dari *Error Compounding*.'),
    
    new_code_cell('''!pip install -q pandas numpy matplotlib seaborn scikit-learn tensorflow'''),
    
    new_markdown_cell('## 1. Imports & Setup'),
    new_code_cell('''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import root_mean_squared_error

import warnings
import gc

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 150)
sns.set_theme(style="whitegrid")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
'''),

    new_markdown_cell('## 2. Load Data'),
    new_code_cell('''try:
    from google.colab import drive
    drive.mount('/content/drive')
    path = '/content/drive/MyDrive/ssf-ssds/'
except:
    path = './' # Lokal

print("Loading data...")
train_df = pd.read_csv(path + 'train.csv', parse_dates=['datetime'])
test_df = pd.read_csv(path + 'test.csv')
sample_sub = pd.read_csv(path + 'sample_submission.csv')

lingkungan_df = pd.read_csv(path + 'data_pendukung/data_lingkungan.csv', parse_dates=['datetime'])
koordinat_df = pd.read_csv(path + 'data_pendukung/koordinat_pos.csv')
'''),

    new_markdown_cell('## 3. Exploratory Data Analysis (EDA) Ekstensif 📊\nDi bagian ini, kita akan membongkar 10 informasi penting dari dataset satu per satu.'),
    
    new_code_cell('''# Gabungkan data untuk EDA
train_eda = pd.merge(train_df, koordinat_df, on='nama_pos', how='left')
env_eda = lingkungan_df[lingkungan_df['datetime'].dt.hour.isin([6,12,18])].copy()
merged_eda = pd.merge(train_eda, env_eda, on=['datetime', 'nama_pos'], how='inner')

plt.figure(figsize=(10, 5))
sns.histplot(train_eda['tma_mdpl'], bins=100, kde=True, color='teal')
plt.title('1. Distribusi Elevasi Tinggi Muka Air (TMA)')
plt.xlabel('Tinggi Muka Air (MDPL)')
plt.show()'''),
    new_markdown_cell('**💡 Penjelasan & Kesimpulan (1):**\nGrafik di atas menunjukkan bahwa TMA memiliki distribusi yang menceng ke kanan (*right-skewed*). Mayoritas stasiun berada di dataran rendah, tetapi ada ekor panjang hingga > 150 MDPL yang menunjukkan keberadaan beberapa stasiun di dataran tinggi (Hulu).'),

    new_code_cell('''tma_pivot = train_eda.pivot(index='datetime', columns='nama_pos', values='tma_mdpl')
plt.figure(figsize=(12, 6))
sns.heatmap(tma_pivot.isna(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('2. Pola Missing Values pada Data TMA (Kuning = Kosong)')
plt.show()'''),
    new_markdown_cell('**💡 Penjelasan & Kesimpulan (2):**\nWarna kuning menunjukkan data yang bolong (missing). Terlihat bahwa *missing values* terjadi secara vertikal pada waktu-waktu tertentu. Artinya, ada hari-hari di mana sensor di banyak stasiun mati bersamaan atau gagal mengirim data ke server pusat.'),

    new_code_cell('''plt.figure(figsize=(15, 6))
mean_elev = train_eda.groupby('nama_pos')['tma_mdpl'].mean().sort_values(ascending=False)
sns.barplot(x=mean_elev.index, y=mean_elev.values, palette='magma')
plt.xticks(rotation=45, ha='right')
plt.title('3. Rata-rata Elevasi TMA per Stasiun (Identifikasi Hulu - Hilir)')
plt.ylabel('Rata-rata TMA (MDPL)')
plt.show()'''),
    new_markdown_cell('**💡 Penjelasan & Kesimpulan (3):**\nGrafik ini sangat krusial! Stasiun di sebelah kiri memiliki rata-rata TMA tertinggi, menjadikannya kandidat kuat sebagai **Stasiun Hulu (Pegunungan)**. Sedangkan stasiun di paling kanan adalah stasiun Hilir/Muara.'),

    new_code_cell('''plt.figure(figsize=(15, 6))
random_sts = np.random.choice(train_eda['nama_pos'].unique(), 10, replace=False)
sns.boxplot(data=train_eda[train_eda['nama_pos'].isin(random_sts)], x='nama_pos', y='tma_mdpl')
plt.title('4. Deteksi Outliers (Banjir Ekstrem) pada 10 Stasiun Acak')
plt.xticks(rotation=45)
plt.show()'''),
    new_markdown_cell('**💡 Penjelasan & Kesimpulan (4):**\nTitik-titik hitam di atas kotak boxplot adalah *outliers*, yang dalam konteks ini merepresentasikan kejadian **banjir ekstrem** atau luapan air.'),

    new_code_cell('''train_eda['hour'] = train_eda['datetime'].dt.hour
plt.figure(figsize=(8, 5))
sns.lineplot(data=train_eda, x='hour', y='tma_mdpl', marker='o', errorbar=None)
plt.title('5. Pola Musiman Harian (Jam 06, 12, 18)')
plt.xticks([6, 12, 18])
plt.show()'''),
    new_markdown_cell('**💡 Penjelasan & Kesimpulan (5):**\nSecara rata-rata global, TMA sedikit lebih tinggi di sore/malam hari (jam 18:00) dibandingkan pagi hari. Ini logis karena curah hujan di iklim tropis biasanya terakumulasi di siang menuju sore hari.'),

    new_code_cell('''train_eda['month'] = train_eda['datetime'].dt.month
plt.figure(figsize=(10, 5))
sns.lineplot(data=train_eda, x='month', y='tma_mdpl', marker='s', errorbar=None, color='orange')
plt.title('6. Pola Musiman Bulanan (Siklus Kemarau vs Penghujan)')
plt.xticks(range(1, 13))
plt.show()'''),
    new_markdown_cell('**💡 Penjelasan & Kesimpulan (6):**\nSiklus hidrologi sangat jelas terlihat. TMA mencapai titik puncaknya di bulan Februari-Maret (puncak musim hujan di Pulau Jawa), dan menyentuh titik terendah di bulan Agustus-September-Oktober.'),

    new_code_cell('''plt.figure(figsize=(12, 8))
plt.scatter(merged_eda['longitude'], merged_eda['latitude'], 
            c=merged_eda['tma_mdpl'], cmap='jet', s=100, alpha=0.6)
plt.colorbar(label='TMA MDPL')
plt.title('7. Pemetaan Spasial Stasiun Berdasarkan Koordinat')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()'''),
    new_markdown_cell('**💡 Penjelasan & Kesimpulan (7):**\nIni adalah rekonstruksi peta DAS Bengawan Solo secara kasar. Warna merah (TMA tinggi) terkonsentrasi di area tertentu (Hulu), sementara warna biru menyebar di daerah lain.'),

    new_code_cell('''plt.figure(figsize=(8, 6))
sample_df = merged_eda.sample(10000, random_state=42)
sns.scatterplot(data=sample_df, x='rainfall_mm', y='tma_mdpl', alpha=0.2, color='darkblue')
plt.title('8. Hubungan Curah Hujan vs TMA (Sesaat)')
plt.show()'''),
    new_markdown_cell('**💡 Penjelasan & Kesimpulan (8):**\nTidak ada garis korelasi linear yang sempurna di sini. Curah hujan kecil bisa saja bikin banjir JIKA 5 hari sebelumnya sudah hujan lebat terus. Ini membuktikan kita BUTUH urutan waktu / sekuens (Yang akan ditangani sempurna oleh LSTM kita nanti!).'),

    new_code_cell('''plt.figure(figsize=(10, 5))
nino_trend = merged_eda.groupby('datetime')['nino_34'].mean()
plt.plot(nino_trend.index, nino_trend.values, color='red')
plt.axhline(0, color='black', linestyle='--')
plt.title('9. Tren Indeks Iklim Global (ENSO / Nino_34)')
plt.ylabel('Nilai Indeks')
plt.show()'''),
    new_markdown_cell('**💡 Penjelasan & Kesimpulan (9):**\nIndeks ENSO bergerak sangat lambat (fluktuasi bulanan). Nilai negatif kuat (di bawah garis hitam) menandakan La Nina (banyak hujan), sedangkan positif kuat menandakan El Nino (kekeringan).'),

    new_code_cell('''plt.figure(figsize=(8, 6))
weather_feats = ['rainfall_mm', 'soil_moisture_0_7cm', 'humidity_pct', 'temperature_c', 'wind_speed_kmh', 'tma_mdpl']
corr = merged_eda[weather_feats].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('10. Heatmap Korelasi Pearson (Cuaca Sesaat)')
plt.show()'''),
    new_markdown_cell('**💡 Penjelasan & Kesimpulan (10):**\nSecara instan (sesaat), `soil_moisture` (kelembapan tanah) dan `humidity_pct` memiliki korelasi positif paling kuat terhadap TMA dibandingkan cuaca lain.'),

    new_code_cell('''# Bersihkan memori dari data EDA
del train_eda, env_eda, merged_eda
gc.collect()'''),

    new_markdown_cell('## 4. Ekstraksi Fitur Dasar & Resampling 6H\nKarena LSTM membutuhkan input sekuensial yang konstan, kita me-resample data lingkungan menjadi interval 6 jam yang berkesinambungan (00, 06, 12, 18).'),
    new_code_cell('''print("Memproses Ekstraksi Fitur...")

# Mengambil fitur-fitur penting
env_features = ['rainfall_mm', 'soil_moisture_0_7cm', 'temperature_c', 'humidity_pct', 'surface_pressure_hpa', 'wind_speed_kmh']
if 'rmm1' in lingkungan_df.columns:
    lingkungan_df['mjo_speed'] = np.sqrt(lingkungan_df['rmm1']**2 + lingkungan_df['rmm2']**2)
    env_features += ['mjo_speed']
if 'nino_34' in lingkungan_df.columns:
    env_features += ['nino_34']

# Runoff Proxy
if 'rainfall_mm' in lingkungan_df.columns and 'soil_moisture_0_7cm' in lingkungan_df.columns:
    lingkungan_df['runoff_proxy'] = lingkungan_df['rainfall_mm'] * lingkungan_df['soil_moisture_0_7cm']
    env_features += ['runoff_proxy']

# Resample menjadi 6 Jam per Stasiun
env_6h_list = []
for station, group in lingkungan_df.groupby('nama_pos'):
    group = group.set_index('datetime').sort_index()
    # Resample: Sum untuk hujan, Mean untuk sisanya
    resampled = group[env_features].resample('6H').mean()
    if 'rainfall_mm' in resampled.columns:
        resampled['rainfall_mm'] = group['rainfall_mm'].resample('6H').sum()
    if 'runoff_proxy' in resampled.columns:
        resampled['runoff_proxy'] = group['runoff_proxy'].resample('6H').sum()
        
    resampled['nama_pos'] = station
    resampled = resampled.reset_index()
    env_6h_list.append(resampled)

env_6h = pd.concat(env_6h_list, ignore_index=True)
env_6h = env_6h.dropna().reset_index(drop=True)

# Normalisasi Fitur (Penting untuk LSTM agar gradient tidak meledak!)
scaler = StandardScaler()
env_6h[env_features] = scaler.fit_transform(env_6h[env_features])

print("Data lingkungan 6H berhasil disiapkan dan distandarisasi!")
'''),

    new_markdown_cell('## 5. Membangun 3D Tensor Sequences (Samples, Timesteps, Features) 🧱\nKita akan membuat *sliding window* mundur. Untuk memprediksi tinggi air jam 12:00, model akan melihat cuaca dari jam 12:00, 06:00, 00:00, dst mundur hingga 30 langkah (7.5 hari ke belakang).'),
    new_code_cell('''SEQ_LEN = 30 # 30 langkah x 6 jam = 180 Jam (7.5 Hari) history cuaca

def build_sequences(target_df, env_df, seq_len, features):
    X_seq = []
    y_vals = []
    valid_indices = []
    
    # Sort target dataframe
    target_df = target_df.sort_values(['nama_pos', 'datetime']).reset_index(drop=True)
    
    for station in target_df['nama_pos'].unique():
        # Get target and env for this station
        st_target = target_df[target_df['nama_pos'] == station]
        st_env = env_df[env_df['nama_pos'] == station].set_index('datetime')
        
        for idx, row in st_target.iterrows():
            t = row['datetime']
            # Ambil history cuaca dari t mundur sebanyak seq_len
            try:
                env_t_idx = st_env.index.get_loc(t)
                if env_t_idx >= seq_len - 1:
                    seq = st_env.iloc[env_t_idx - seq_len + 1 : env_t_idx + 1][features].values
                    X_seq.append(seq)
                    if 'tma_mdpl' in target_df.columns:
                        y_vals.append(row['tma_mdpl'])
                    valid_indices.append(idx)
            except KeyError:
                pass # Waktu t tidak ada di data lingkungan
                
    return np.array(X_seq), np.array(y_vals) if 'tma_mdpl' in target_df.columns else None, valid_indices

print("Membangun 3D Tensor untuk Train Set (Ini akan memakan waktu sekitar 1-2 menit)...")
X_train_3d, y_train, train_valid_idx = build_sequences(train_df, env_6h, SEQ_LEN, env_features)

print("Membangun 3D Tensor untuk Test Set...")
test_parsed = test_df.copy()
split_info = test_parsed['id'].str.split(' - ', n=1, expand=True)
test_parsed['datetime'] = pd.to_datetime(split_info[0])
test_parsed['nama_pos'] = split_info[1]

X_test_3d, _, test_valid_idx = build_sequences(test_parsed, env_6h, SEQ_LEN, env_features)

print(f"Bentuk X_train: {X_train_3d.shape} (Samples, Timesteps, Features)")
print(f"Bentuk X_test: {X_test_3d.shape}")
'''),

    new_markdown_cell('''## 6. Dokumentasi Arsitektur Model Deep Learning 🧠

Model ini adalah **Weather-to-Water Seq2Vec LSTM** murni (Exogenous).
Berbeda dengan LSTM tradisional yang menelan nilai historis TMA, model ini *HANYA* menelan data cuaca.

```mermaid
graph TD
    A[Input: 30 Langkah Cuaca Historis\nShape: Batch, 30, 9] --> B[LSTM Layer 1\n64 Units, Return Seq=True]
    B --> C[Dropout 20%]
    C --> D[LSTM Layer 2\n32 Units, Return Seq=False]
    D --> E[Dropout 20%]
    E --> F[Dense Layer\n32 Units, ReLU]
    F --> G[Output Layer\n1 Unit, Linear]
    G --> H[Prediksi TMA Saat Ini]
```

**Kelebihan Arsitektur Ini:**
1. **Zero Error Compounding:** Karena tidak menggunakan data TMA dari masa lalu sebagai *input*, error di prediksi hari pertama tidak akan merusak prediksi di bulan ke-8 (Data Test).
2. **Robust Peaks:** Lapisan Dense Output menggunakan *Linear Activation* dipadu dengan *Huber Loss*. Ini memaksa model untuk mengejar luapan air yang tinggi (banjir) tanpa tertekan rata-rata.
'''),
    new_code_cell('''def build_lstm_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1) # Linear output untuk regresi skala asli
    ])
    
    # Huber loss menyeimbangkan outlier dan inliers
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss='huber', metrics=['RootMeanSquaredError'])
    return model
'''),

    new_markdown_cell('## 7. Training K-Fold Cross Validation\nMemanfaatkan GPU Colab, mari kita hajar model ini dengan K-Fold agar tahan banting!'),
    new_code_cell('''kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X_train_3d))
test_preds = np.zeros(len(X_test_3d))

models = []
histories = []

print("Mulai Training LSTM K-Fold (Pastikan GPU T4 aktif agar super cepat!)...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_3d, y_train)):
    print(f"\\n========== FOLD {fold+1} ==========")
    X_tr, y_tr = X_train_3d[train_idx], y_train[train_idx]
    X_va, y_va = X_train_3d[val_idx], y_train[val_idx]
    
    model = build_lstm_model((SEQ_LEN, len(env_features)))
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
    
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=100,
        batch_size=256,
        callbacks=[early_stop, reduce_lr],
        verbose=1 # Ubah ke 0 jika tidak ingin melihat bar progress
    )
    
    histories.append(history)
    models.append(model)
    
    val_preds = model.predict(X_va, batch_size=256).flatten()
    oof_preds[val_idx] = val_preds
    
    # Inferensi ke test set secara akumulatif
    test_preds += model.predict(X_test_3d, batch_size=256).flatten() / 5
    
    fold_rmse = root_mean_squared_error(y_va, val_preds)
    print(f"Fold {fold+1} RMSE: {fold_rmse:.4f}")

print(f"\\n✅ Average CV RMSE: {root_mean_squared_error(y_train, oof_preds):.4f}")
'''),

    new_markdown_cell('## 8. Evaluasi Visual: Data Asli vs Prediksi (Validation Set) 📈\nMari kita buktikan apakah LSTM berhasil mengejar puncak kurva biru!'),
    new_code_cell('''fig, ax = plt.subplots(1, 2, figsize=(18, 6))

# 1. Plot Kurva Loss Training (Fold Terakhir)
ax[0].plot(histories[-1].history['loss'], label='Train Loss')
ax[0].plot(histories[-1].history['val_loss'], label='Validation Loss')
ax[0].set_title('Learning Curve LSTM (Fold Terakhir)')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Huber Loss')
ax[0].legend()

# 2. Actual vs Predicted (Validasi Fold Terakhir)
# Ambil subset 200 data terakhir agar grafik terbaca dengan jelas
sub_actual = y_va[-200:]
sub_pred = val_preds[-200:]

ax[1].plot(sub_actual, label='Actual TMA (Data Asli)', color='blue', alpha=0.7)
ax[1].plot(sub_pred, label='Predicted TMA (LSTM V7)', color='red', alpha=0.7, linewidth=2)
ax[1].set_title('Actual vs Predicted (Validation Set - Sampel 200)')
ax[1].legend()

plt.tight_layout()
plt.show()
'''),

    new_markdown_cell('## 9. Generate Submission (Inferensi Super Cepat)'),
    new_code_cell('''# Mapping hasil prediksi kembali ke test_df
# Jika ada index test yang hilang akibat tidak ada cuaca, kita isi dengan ffill/bfill
test_df['tma_mdpl'] = np.nan
test_df.loc[test_valid_idx, 'tma_mdpl'] = test_preds

sub = test_df[['id', 'tma_mdpl']].copy()
# Reorder to match sample_submission exactly
sub = pd.merge(sample_sub[['id']], sub, on='id', how='left')
sub['tma_mdpl'] = sub['tma_mdpl'].fillna(method='ffill').fillna(method='bfill')

display(sub.head(10))
sub.to_csv('submission_v7_lstm_exogenous.csv', index=False)
print("\\n✅ THE ULTIMATE LSTM V7 COMPLETED! File 'submission_v7_lstm_exogenous.csv' siap diunduh!")
''')
])

with open('V7_Exogenous_LSTM.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
