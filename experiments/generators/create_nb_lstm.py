import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()

nb.cells.extend([
    new_markdown_cell('# 🧠 Deep Learning LSTM untuk Forecasting TMA (Sultan Edition) 👑\n\nNotebook ini dirancang khusus untuk mengatasi masalah *error compounding* pada *recursive forecasting* yang dialami oleh model XGBoost. Menggunakan arsitektur **Multivariate Spatial LSTM** (Keras/TensorFlow), model ini memprediksi seluruh 30 stasiun secara bersamaan (Simultaneous Multi-Output), sehingga secara implisit mempelajari jaringan spasial sungai.\n\nNotebook ini juga dilengkapi dengan 10 Visualisasi Exploratory Data Analysis (EDA) mendalam!'),
    
    new_code_cell('''# Instalasi library tambahan (jika belum ada)
!pip install -q pandas numpy matplotlib seaborn scikit-learn tensorflow
'''),

    new_markdown_cell('## 1. Imports & Setup'),
    new_code_cell('''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import gc

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# Memastikan TF menggunakan GPU jika tersedia (T4 di Colab)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
'''),

    new_markdown_cell('## 2. Load Data'),
    new_code_cell('''# Mount Google Drive (hanya jalan di Colab)
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("Google Drive mounted successfully!")
except:
    print("Not running in Google Colab, skipping drive mount.")

# Path ke folder ssf-ssds di root Google Drive Anda
path = '/content/drive/MyDrive/ssf-ssds/'
# Jika di lokal, ganti jadi path = './'

print("Loading data...")
train_df = pd.read_csv(path + 'train.csv', parse_dates=['datetime'])
test_df = pd.read_csv(path + 'test.csv')
sample_sub = pd.read_csv(path + 'sample_submission.csv')

lingkungan_df = pd.read_csv(path + 'data_pendukung/data_lingkungan.csv', parse_dates=['datetime'])
koordinat_df = pd.read_csv(path + 'data_pendukung/koordinat_pos.csv')

print(f'Train shape: {train_df.shape}')
print(f'Lingkungan shape: {lingkungan_df.shape}')
'''),

    new_markdown_cell('## 3. Ekstensif Exploratory Data Analysis (EDA) - 10 Insights 📊\n\nKita akan menggali anomali, tren spasial, dan korelasi dari data TMA dan Lingkungan.'),
    new_code_cell('''# Siapkan data untuk plot (Ambil subset agar tidak terlalu berat)
train_eda = train_df.copy()
env_eda = lingkungan_df[lingkungan_df['datetime'].dt.hour.isin([6,12,18])].copy()
merged_eda = pd.merge(train_eda, env_eda, on=['datetime', 'nama_pos'], how='inner')

# Mengatur figure grid besar
fig = plt.figure(figsize=(24, 30))
fig.suptitle('Exploratory Data Analysis (EDA) - 10 Insights Utama', fontsize=24, y=0.92)

# 1. Distribusi TMA Keseluruhan
ax1 = plt.subplot(5, 2, 1)
sns.histplot(train_eda['tma_mdpl'], bins=100, kde=True, ax=ax1, color='skyblue')
ax1.set_title('1. Distribusi Global TMA (Menceng ke Kanan)')

# 2. Rata-rata Elevasi TMA per Stasiun (Proxy Hulu-Hilir)
ax2 = plt.subplot(5, 2, 2)
mean_tma = train_eda.groupby('nama_pos')['tma_mdpl'].mean().sort_values(ascending=False)
sns.barplot(x=mean_tma.values, y=mean_tma.index, ax=ax2, palette='viridis')
ax2.set_title('2. Rata-rata TMA per Stasiun (Hulu vs Hilir)')
ax2.set_xlabel('TMA MDPL')

# 3. Heatmap TMA antar Stasiun (Spatial Correlation)
ax3 = plt.subplot(5, 2, 3)
tma_pivot = train_eda.pivot(index='datetime', columns='nama_pos', values='tma_mdpl')
sns.heatmap(tma_pivot.corr(), cmap='coolwarm', cbar=True, ax=ax3, xticklabels=False, yticklabels=False)
ax3.set_title('3. Korelasi Spasial Antar Stasiun (Warna Merah = Saling Mempengaruhi)')

# 4. Outlier Analysis (Boxplot 10 Stasiun Acak)
ax4 = plt.subplot(5, 2, 4)
random_stations = np.random.choice(train_eda['nama_pos'].unique(), 10, replace=False)
sns.boxplot(data=train_eda[train_eda['nama_pos'].isin(random_stations)], x='nama_pos', y='tma_mdpl', ax=ax4)
ax4.set_title('4. Analisis Anomali & Ekstrem TMA (Outliers)')
ax4.tick_params(axis='x', rotation=45)

# 5. Missing Value Pattern
ax5 = plt.subplot(5, 2, 5)
sns.heatmap(tma_pivot.isna(), cbar=False, cmap='viridis', ax=ax5, yticklabels=False, xticklabels=False)
ax5.set_title('5. Pola Missing Value TMA (Kuning = Data Hilang)')

# 6. Seasonality Jam (Harian)
ax6 = plt.subplot(5, 2, 6)
train_eda['hour'] = train_eda['datetime'].dt.hour
sns.lineplot(data=train_eda, x='hour', y='tma_mdpl', ax=ax6, marker='o', color='purple')
ax6.set_title('6. Tren TMA Berdasarkan Jam (06:00, 12:00, 18:00)')

# 7. Seasonality Bulanan (Hujan vs Kemarau)
ax7 = plt.subplot(5, 2, 7)
train_eda['month'] = train_eda['datetime'].dt.month
sns.lineplot(data=train_eda, x='month', y='tma_mdpl', ax=ax7, marker='s', color='orange')
ax7.set_title('7. Tren Musiman Bulanan (Siklus Hujan-Kemarau)')

# 8. Scatter Plot Hujan vs TMA (Ambil sampel 5000 titik agar tidak nge-lag)
ax8 = plt.subplot(5, 2, 8)
sample_scatter = merged_eda.sample(5000, random_state=42)
sns.scatterplot(data=sample_scatter, x='rainfall_mm', y='tma_mdpl', alpha=0.3, ax=ax8)
ax8.set_title('8. Korelasi Titik Hujan vs TMA')

# 9. Tren El Nino (nino_34) seiring waktu
ax9 = plt.subplot(5, 2, 9)
nino_trend = merged_eda.groupby('datetime')['nino_34'].mean()
ax9.plot(nino_trend.index, nino_trend.values, color='red')
ax9.set_title('9. Pergerakan Indeks Iklim Global (Nino_34)')

# 10. Korelasi Heatmap 6 Fitur Cuaca Utama vs TMA
ax10 = plt.subplot(5, 2, 10)
weather_feats = ['rainfall_mm', 'soil_moisture_0_7cm', 'humidity_pct', 'temperature_c', 'wind_speed_kmh', 'tma_mdpl']
sns.heatmap(merged_eda[weather_feats].corr(), annot=True, cmap='Blues', ax=ax10)
ax10.set_title('10. Korelasi Fitur Lingkungan Langsung ke TMA')

plt.tight_layout()
fig.subplots_adjust(top=0.95)
plt.show()

# Bersihkan RAM
del train_eda, env_eda, merged_eda, tma_pivot
gc.collect()
'''),

    new_markdown_cell('## 4. LSTM Data Preprocessing & Pivoting (Float32)\n\nUntuk model Multivariate LSTM, kita mengubah format data dari *long* menjadi *wide* (Pivoting). Baris = Timesteps (Waktu), Kolom = Kombinasi 30 Stasiun & Fitur Cuaca.'),
    new_code_cell('''# 1. Pivot Train TMA
print("Pivoting data target...")
train_df = train_df.dropna(subset=['tma_mdpl']) # Buang missing target jika ada
y_pivot = train_df.pivot(index='datetime', columns='nama_pos', values='tma_mdpl').astype('float32')

# Simpan list 30 stasiun sesuai urutan agar saat inference tidak salah tempat
STATIONS = list(y_pivot.columns)
print(f"Total Stations: {len(STATIONS)}")

# 2. Proses Fitur Lingkungan
print("Memproses fitur lingkungan (Agregasi ke 6 Jam)...")
lingkungan_df = lingkungan_df.sort_values(['nama_pos', 'datetime']).reset_index(drop=True)
features_to_use = ['rainfall_mm', 'soil_moisture_0_7cm', 'humidity_pct', 'temperature_c', 'surface_pressure_hpa']

# Kita lakukan agregasi (mean 6 jam terakhir) 
grouped_env = lingkungan_df.groupby('nama_pos')
for f in features_to_use:
    lingkungan_df[f'{f}_6h'] = grouped_env[f].transform(lambda x: x.rolling(6, min_periods=1).mean())

env_6h = lingkungan_df[['datetime', 'nama_pos'] + [f'{f}_6h' for f in features_to_use]]

# Filter HANYA pada jam pencatatan (06:00, 12:00, 18:00)
env_6h = env_6h[env_6h['datetime'].dt.hour.isin([6, 12, 18])].reset_index(drop=True)

# 3. Pivot Lingkungan menjadi Wide Format (150 Kolom)
print("Pivoting data lingkungan...")
X_env_pivot = env_6h.pivot(index='datetime', columns='nama_pos', values=[f'{f}_6h' for f in features_to_use])
# Ratakan MultiIndex columns
X_env_pivot.columns = [f'{col[0]}_{col[1]}' for col in X_env_pivot.columns]
X_env_pivot = X_env_pivot.astype('float32')

# 4. Gabungkan X (Env) dan Y (Target) berdasarkan datetime
master_df = y_pivot.join(X_env_pivot, how='inner')

# Imputasi jika ada data kosong (forward fill lalu backward fill)
master_df = master_df.fillna(method='ffill').fillna(method='bfill')

print(f"Bentuk Data Master: {master_df.shape} (Baris Waktu, Total Kolom Y + X)")
'''),

    new_code_cell('''# Scaling Data
print("Melakukan Normalisasi (MinMaxScaler)...")
# Pisahkan kolom Y (Target) dan X (Fitur)
y_cols = STATIONS
x_cols = [c for c in master_df.columns if c not in y_cols]

# Kita scale Y dan X secara terpisah agar mudah invers transformasi saat inferensi
scaler_y = MinMaxScaler(feature_range=(0, 1))
scaler_x = MinMaxScaler(feature_range=(0, 1))

master_y_scaled = scaler_y.fit_transform(master_df[y_cols])
master_x_scaled = scaler_x.fit_transform(master_df[x_cols])

# Menyusun dataset untuk LSTM
# Fitur di waktu t: [ Y(t-1), X(t) ]
# Target di waktu t: Y(t)

SEQ_LEN = 14 # Panjang sekuens historis (14 x 6 jam = 3.5 hari)

X_lstm = []
Y_lstm = []
timestamps_lstm = []

master_datetimes = master_df.index.values

for i in range(SEQ_LEN, len(master_y_scaled)):
    # Y(t-15) sampai Y(t-1)
    y_hist = master_y_scaled[i-SEQ_LEN : i]
    # X(t-14) sampai X(t)
    x_hist = master_x_scaled[i-SEQ_LEN+1 : i+1]
    
    # Gabungkan fitur
    combined_hist = np.concatenate([y_hist, x_hist], axis=-1)
    
    X_lstm.append(combined_hist)
    Y_lstm.append(master_y_scaled[i])
    timestamps_lstm.append(master_datetimes[i])
    
X_lstm = np.array(X_lstm, dtype=np.float32)
Y_lstm = np.array(Y_lstm, dtype=np.float32)

print(f"X_lstm shape: {X_lstm.shape}") # (Samples, Timesteps, Features)
print(f"Y_lstm shape: {Y_lstm.shape}") # (Samples, 30)
'''),

    new_markdown_cell('## 5. Arsitektur LSTM Deep Learning 🧠'),
    new_code_cell('''# Split Train dan Validation (80% Train, 20% Val)
split_idx = int(len(X_lstm) * 0.8)
X_train, X_val = X_lstm[:split_idx], X_lstm[split_idx:]
y_train, y_val = Y_lstm[:split_idx], Y_lstm[split_idx:]

print(f"Training data: {X_train.shape}")
print(f"Validation data: {X_val.shape}")

# Membangun Model Sequential
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(128, return_sequences=True, activation='tanh'),
    Dropout(0.2),
    LSTM(64, activation='tanh'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(len(STATIONS), activation='linear') # Output sebanyak 30 stasiun
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
model.compile(optimizer=optimizer, loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])

model.summary()
'''),

    new_markdown_cell('## 6. Training, Callbacks & Evaluasi'),
    new_code_cell('''# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1)
checkpoint = ModelCheckpoint('best_lstm_model.keras', monitor='val_rmse', save_best_only=True, verbose=0)

print("Memulai Training LSTM...")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)
'''),

    new_code_cell('''# Plot Loss dan RMSE
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

ax[0].plot(history.history['loss'], label='Train Loss (MSE)')
ax[0].plot(history.history['val_loss'], label='Val Loss (MSE)')
ax[0].set_title('Model Loss Over Epochs')
ax[0].legend()

ax[1].plot(history.history['rmse'], label='Train RMSE')
ax[1].plot(history.history['val_rmse'], label='Val RMSE')
ax[1].set_title('Model RMSE Over Epochs')
ax[1].legend()

plt.show()

# Load best model
model = tf.keras.models.load_model('best_lstm_model.keras')
'''),

    new_markdown_cell('## 7. Inference & Recursive Generation (Test Set) 🚀\n\nKita mengekstraksi tanggal unik di Test Set, menyiapkan X (Lingkungan masa depan) dan memakai output prediksi waktu *t* untuk masuk sebagai Y historis di prediksi waktu *t+1*.'),
    new_code_cell('''# Parse Test Datetimes
test_df['datetime'] = pd.to_datetime(test_df['id'].str.split(' - ', n=1, expand=True)[0])
test_datetimes = sorted(test_df['datetime'].unique())
print(f"Total langkah inferensi: {len(test_datetimes)}")

# 1. Siapkan data lingkungan untuk semua test datetimes
# Kita gunakan X_env_pivot yang sudah dibuat. Namun kita butuh X_env untuk test_datetimes.
# Mari kita buat pivot keseluruhan environment (termasuk masa depan)
print("Pivoting lingkungan untuk inferensi...")
env_6h_all = lingkungan_df[['datetime', 'nama_pos'] + [f'{f}_6h' for f in features_to_use]]
env_6h_all = env_6h_all[env_6h_all['datetime'].dt.hour.isin([6, 12, 18])].reset_index(drop=True)

X_env_all_pivot = env_6h_all.pivot(index='datetime', columns='nama_pos', values=[f'{f}_6h' for f in features_to_use])
X_env_all_pivot.columns = [f'{col[0]}_{col[1]}' for col in X_env_all_pivot.columns]
X_env_all_pivot = X_env_all_pivot.astype('float32').fillna(method='ffill').fillna(method='bfill')

# Scale menggunakan scaler yang sama
X_env_all_scaled = pd.DataFrame(scaler_x.transform(X_env_all_pivot), index=X_env_all_pivot.index, columns=X_env_all_pivot.columns)

# 2. Inisialisasi Sequence Terakhir dari Training sebagai Buffer Memori
last_train_dt = master_df.index[-1]
# Buffer untuk Y (Target TMA Terakhir) -> (SEQ_LEN, 30)
buffer_y = master_y_scaled[-SEQ_LEN:]

# Dictionary untuk menyimpan prediksi akhir
predictions_dict = {}

print("Memulai Recursive Loop LSTM...")
for i, dt in enumerate(test_datetimes):
    # Ambil fitur lingkungan dari t-13 sampai t
    # Jika dt tidak ada di lingkungan (anomali), gunakan baris terakhir yg tersedia
    end_idx = X_env_all_scaled.index.get_loc(dt, method='pad')
    # Ambil 14 data terakhir lingkungan
    buffer_x = X_env_all_scaled.iloc[end_idx - SEQ_LEN + 1 : end_idx + 1].values
    
    # Pastikan shape (14, 150)
    if len(buffer_x) < SEQ_LEN:
        pad_size = SEQ_LEN - len(buffer_x)
        padding = np.repeat([buffer_x[0]], pad_size, axis=0)
        buffer_x = np.vstack([padding, buffer_x])
        
    # Gabungkan fitur: buffer_y dan buffer_x
    # Shape buffer_y: (14, 30), Shape buffer_x: (14, 150)
    current_input = np.concatenate([buffer_y, buffer_x], axis=-1)
    
    # Prediksi untuk waktu t
    # Tambah dimensi batch: (1, 14, 180)
    current_input = np.expand_dims(current_input, axis=0)
    pred_y_scaled = model.predict(current_input, verbose=0) # (1, 30)
    
    # Simpan Prediksi Scaled ke Buffer (Buang yg tertua, masukkan yg baru)
    buffer_y = np.vstack([buffer_y[1:], pred_y_scaled[0]])
    
    # Invers Transform ke nilai asli TMA
    pred_y_asli = scaler_y.inverse_transform(pred_y_scaled)[0]
    
    # Masukkan ke dictionary
    for idx_st, st in enumerate(STATIONS):
        id_str = f"{dt.strftime('%Y-%m-%d %H:%M:%S')} - {st}"
        predictions_dict[id_str] = pred_y_asli[idx_st]
        
    if (i+1) % 50 == 0:
        print(f"Langkah {i+1} dari {len(test_datetimes)} Selesai...")

print("Inferensi Selesai!")
'''),

    new_markdown_cell('## 8. Export to CSV'),
    new_code_cell('''# Convert dictionary to dataframe
sub_pred = pd.DataFrame(list(predictions_dict.items()), columns=['id', 'tma_mdpl'])

# Cocokkan urutan dengan sample submission
sub = pd.merge(sample_sub[['id']], sub_pred, on='id', how='left')

# Interpolasi jika ada yang bolong
sub['tma_mdpl'] = sub['tma_mdpl'].fillna(method='bfill')

print("Sample Submission (First 10 Rows):")
display(sub.head(10))

sub.to_csv('submission_lstm_tma.csv', index=False)
print("\\n✅ File 'submission_lstm_tma.csv' (LSTM) berhasil diekspor!")
''')
])

with open('LSTM_Advanced_TMA.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
