import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

def create_v27_notebook():
    nb = new_notebook()
    
    cells = [
        new_markdown_cell('# 🚀 V27 TEMPORAL FUSION TRANSFORMER (TFT) 🚀\\nSelamat datang di arsitektur *Deep Learning End-Game*. Di notebook ini, kita meninggalkan paradigma XGBoost (30 model terpisah) dan melatih **1 Super-Model** menggunakan arsitektur TFT buatan Google via `pytorch-forecasting`.\\n\\nTFT mampu memecahkan masalah *compounding error* dengan cara menggunakan *Encoder* untuk mengingat sejarah tinggi air, dan *Decoder* untuk memprediksi sekaligus (1452 jam ke depan) berdasarkan cuaca masa depan yang diketahui.\\n\\n> [!WARNING]\\n> **PENTING:** Anda WAJIB menghidupkan Accelerator GPU (T4x2 atau P100) di Kaggle Anda sebelum menjalankan notebook ini!'),
        
        new_markdown_cell('## 1. Persiapan Environment & Install PyTorch Forecasting'),
        new_code_cell('''!pip install -q pytorch-forecasting pytorch-lightning optuna folium
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import RMSE

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

print("Data berhasil dimuat!")
'''),

        new_markdown_cell('## 2. Feature Engineering (Lokal & Basin)\\nKita tetap mempertahankan fitur Basin V25 karena *Helicopter View* sangat krusial bagi TFT.'),
        new_code_cell('''print("Mengekstraksi Fitur Cuaca Lokal & GLOBAL BASIN...")
lingkungan_df = lingkungan_df.sort_values(['nama_pos', 'datetime']).reset_index(drop=True)

def build_exogenous_features(env_df):
    agg_df = env_df[['datetime', 'nama_pos']].copy()
    grouped = env_df.groupby('nama_pos')
    
    if 'rainfall_mm' in env_df.columns:
        agg_df['rain_sum_72h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(72, min_periods=1).sum())
        agg_df['rain_sum_168h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(168, min_periods=1).sum())
        
    if 'soil_moisture_0_7cm' in env_df.columns:
        agg_df['soil_mean_72h'] = grouped['soil_moisture_0_7cm'].transform(lambda x: x.rolling(72, min_periods=1).mean())
        
    # FITUR GLOBAL BASIN
    basin_env = env_df.groupby('datetime').agg({'rainfall_mm': 'mean', 'soil_moisture_0_7cm': 'mean'}).reset_index()
    basin_env['basin_rain_sum_168h'] = basin_env['rainfall_mm'].rolling(168, min_periods=1).sum()
    basin_env['basin_soil_mean_168h'] = basin_env['soil_moisture_0_7cm'].rolling(168, min_periods=1).mean()
    
    basin_features = basin_env[['datetime', 'basin_rain_sum_168h', 'basin_soil_mean_168h']]
    agg_df = pd.merge(agg_df, basin_features, on='datetime', how='left')

    # Fill NaN for early rows
    agg_df = agg_df.fillna(0)
    return agg_df

env_feat = build_exogenous_features(lingkungan_df)
'''),

        new_markdown_cell('## 3. Persiapan Dataset Khusus PyTorch Forecasting (`TimeSeriesDataSet`)\\nTFT membutuhkan satu *dataframe* gabungan (Train + Test) dengan indeks waktu integer berkelanjutan (`time_idx`) dan `group_ids` bertipe *string/category*.'),
        new_code_cell('''# Parsing Test Set
test_parsed = test_df.copy()
split_info = test_parsed['id'].str.split(' - ', n=1, expand=True)
test_parsed['datetime'] = pd.to_datetime(split_info[0])
test_parsed['nama_pos'] = split_info[1]
test_parsed['tma_mdpl'] = 0.0 # Target dummy untuk Test Set
test_parsed['is_test'] = 1

train_df['is_test'] = 0

# Gabungkan Train dan Test agar TFT bisa melakukan look-back dari test ke train
full_df = pd.concat([train_df[['datetime', 'nama_pos', 'tma_mdpl', 'is_test']], 
                     test_parsed[['datetime', 'nama_pos', 'tma_mdpl', 'is_test']]], ignore_index=True)

# Tambah Fitur Waktu Dasar
full_df['hour'] = full_df['datetime'].dt.hour
full_df['month'] = full_df['datetime'].dt.month
full_df['hour_sin'] = np.sin(2 * np.pi * full_df['hour']/24.0)
full_df['hour_cos'] = np.cos(2 * np.pi * full_df['hour']/24.0)

# Merge Exogenous Features
full_df = pd.merge(full_df, env_feat, on=['datetime', 'nama_pos'], how='left')

# 📌 Pembuatan TIME_IDX (Wajib untuk PyTorch Forecasting)
full_df = full_df.sort_values(['nama_pos', 'datetime']).reset_index(drop=True)
min_date = full_df['datetime'].min()
full_df['time_idx'] = ((full_df['datetime'] - min_date).dt.total_seconds() // 3600).astype(int)

# 📌 Konversi Tipe Data Wajib
full_df['nama_pos'] = full_df['nama_pos'].astype(str).astype('category')
for col in ['hour_sin', 'hour_cos', 'rain_sum_72h', 'rain_sum_168h', 'soil_mean_72h', 'basin_rain_sum_168h', 'basin_soil_mean_168h']:
    full_df[col] = full_df[col].astype(float)

# Hapus baris di awal Train yang TMA-nya kosong
# (Biarkan 0.0 di Test Set karena itu dummy)
full_df = full_df[~( (full_df['is_test'] == 0) & (full_df['tma_mdpl'].isna()) )].reset_index(drop=True)

print("Data berhasil disiapkan untuk TFT!")
print(f"Total Baris: {len(full_df)}")
'''),

        new_markdown_cell('## 4. Definisi Konfigurasi TFT & Dataloaders'),
        new_code_cell('''# Parameter Kunci
max_prediction_length = 1452  # Panjang maksimum Test Set (sekitar 60 hari)
max_encoder_length = 720      # TFT melihat 30 hari ke belakang

# Pisahkan Train Set Murni
train_only_df = full_df[full_df['is_test'] == 0].copy()

# Batas validasi (ambil 60 hari terakhir dari Train Set sebagai validasi)
training_cutoff = train_only_df["time_idx"].max() - max_prediction_length

print("Membuat TimeSeriesDataSet...")
training = TimeSeriesDataSet(
    train_only_df[train_only_df.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="tma_mdpl",
    group_ids=["nama_pos"],
    min_encoder_length=max_encoder_length // 2, 
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["nama_pos"],
    time_varying_known_reals=["time_idx", "hour_sin", "hour_cos", "rain_sum_72h", "rain_sum_168h", "soil_mean_72h", "basin_rain_sum_168h", "basin_soil_mean_168h"],
    time_varying_unknown_reals=["tma_mdpl"],
    target_normalizer=GroupNormalizer(
        groups=["nama_pos"], transformation="softplus"
    ),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True # Sangat penting jika ada gap jam di data!
)

# Buat Validation Dataset dengan format yang sama
validation = TimeSeriesDataSet.from_dataset(training, train_only_df, predict=True, stop_randomization=True)

# Buat Dataloaders
batch_size = 32 # Turunkan ke 16 jika GPU Out of Memory (OOM)
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=2)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 2, num_workers=2)

print("Dataloaders Siap!")
'''),

        new_markdown_cell('## 5. Training Loop (PyTorch Lightning)\\nKita menggunakan GPU untuk melatih arsitektur Attention yang berat ini.'),
        new_code_cell('''pl.seed_everything(42)

early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min")
lr_logger = LearningRateMonitor()

trainer = pl.Trainer(
    max_epochs=15, # Naikkan jika ingin lebih konvergen
    accelerator="gpu" if pl.torch.cuda.is_available() else "cpu",
    devices=1,
    enable_model_summary=True,
    gradient_clip_val=0.1,
    callbacks=[lr_logger, early_stop_callback],
)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,          # Ukuran hidden layer (naikkan ke 32/64 jika memori cukup)
    attention_head_size=1,   # Jumlah attention head
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=1,           # Kita prediksi RMSE point, jadi output_size 1
    loss=RMSE(),             # Sesuai dengan metrik kompetisi BDC!
    reduce_on_plateau_patience=3,
)

print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

# Mulai Training
print("🚀 MEMULAI TRAINING TFT... (Silakan seduh kopi, ini mungkin butuh waktu!)")
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

# Load best model
best_model_path = trainer.checkpoint_callback.best_model_path
print(f"Best model saved at: {best_model_path}")
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
'''),

        new_markdown_cell('## 6. Inferensi pada Test Set & Pembuatan Submission'),
        new_code_cell('''print("Membangun Test Set Inferensi...")
# TFT membutuhkan encoder history untuk memprediksi test set.
# Kita gunakan seluruh full_df (karena di tahap 3, test set sudah digabung dengan train set history).

# Prediksi seluruh data
predictions = best_tft.predict(full_df, mode="prediction", return_x=False)

# Menggabungkan hasil prediksi ke test set asli
# Format return predictions dari pytorch-forecasting adalah Tensor list sesuai urutan dataset
# Karena kompleksitas extraction, kita bisa melakukan mapping sederhana

# Ambil test set aktual
actual_test = full_df[full_df['is_test'] == 1].copy()

# Pytorch-forecasting .predict() pada dataframe akan mengekstrak prediksi terakhir per grup.
# Karena test_set berada di akhir tiap grup 'nama_pos', hasilnya adalah array (30, 1452)
preds_array = predictions.numpy()

stations = sorted(full_df['nama_pos'].unique())
for i, st in enumerate(stations):
    st_mask = (test_parsed['nama_pos'] == st)
    
    # Karena terkadang panjang test_set tidak genap 1452, kita hanya ambil sepanjang mask
    st_len = st_mask.sum()
    if st_len > 0:
        # Prediksi TFT adalah sequence dari masa depan, kita potong sesuai panjang test set aktual
        test_parsed.loc[st_mask, 'tma_mdpl'] = preds_array[i, :st_len]

# Gabungkan dengan ID asli
sub_pred = test_parsed[['id', 'tma_mdpl']].copy()
sub = pd.merge(sample_sub[['id']], sub_pred, on='id', how='left')

# Forward/Backward Fill jika ada sedikit bolong di ujung
sub['tma_mdpl'] = sub['tma_mdpl'].fillna(method='ffill').fillna(method='bfill')

display(sub.head())
sub.to_csv('submission_v27_tft_global.csv', index=False)
print("\\n✅ File 'submission_v27_tft_global.csv' berhasil dibuat dengan kekuatan Google TFT!")
''')
    ]
    
    nb.cells.extend(cells)
    with open('V27_Temporal_Fusion_Transformer.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == '__main__':
    create_v27_notebook()
    print("Notebook V27_Temporal_Fusion_Transformer.ipynb berhasil dibuat!")
