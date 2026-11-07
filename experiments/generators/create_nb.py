import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()

nb.cells.extend([
    new_markdown_cell('# 🌊 Prediksi Tinggi Muka Air (TMA) BBWS Bengawan Solo\n\nNotebook ini berisi pipeline lengkap untuk kompetisi Prediksi TMA menggunakan XGBoost. Pipeline mencakup EDA, Feature Engineering mendalam, Training dengan K-Fold & Early Stopping, serta Inferensi.'),
    new_markdown_cell('## 1. Imports & Setup'),
    new_code_cell('''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error
import warnings
import gc

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)'''),
    
    new_markdown_cell('## 2. Load Data\nPastikan untuk menjalankan cell mount drive di bawah jika Anda menggunakan Google Colab.'),
    new_code_cell('''# Mount Google Drive (hanya jalan di Colab)
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("Google Drive mounted successfully!")
except:
    print("Not running in Google Colab, skipping drive mount.")

# Path ke folder ssf-ssds di root Google Drive Anda
path = '/content/drive/MyDrive/ssf-ssds/'

# Load data utama
train_df = pd.read_csv(path + 'train.csv', parse_dates=['datetime'])
test_df = pd.read_csv(path + 'test.csv')
sample_sub = pd.read_csv(path + 'sample_submission.csv')

# Load data lingkungan & koordinat
lingkungan_df = pd.read_csv(path + 'data_pendukung/data_lingkungan.csv', parse_dates=['datetime'])
koordinat_df = pd.read_csv(path + 'data_pendukung/koordinat_pos.csv')

print(f'Train shape: {train_df.shape}')
print(f'Test shape: {test_df.shape}')
print(f'Lingkungan shape: {lingkungan_df.shape}')'''),
    
    new_markdown_cell('## 3. Exploratory Data Analysis (EDA)'),
    new_code_cell('''# Cek rentang waktu
print(\"Train Data Range:\", train_df['datetime'].min(), \"to\", train_df['datetime'].max())

# Visualisasi distribusi target TMA
plt.figure(figsize=(10, 5))
sns.histplot(train_df['tma_mdpl'], bins=100, kde=True)
plt.title('Distribusi Tinggi Muka Air (TMA)')
plt.show()

# Visualisasi tren TMA untuk beberapa stasiun
sample_stations = train_df['nama_pos'].unique()[:3]
plt.figure(figsize=(15, 6))
for station in sample_stations:
    temp = train_df[train_df['nama_pos'] == station].set_index('datetime')
    plt.plot(temp.index, temp['tma_mdpl'], label=station, alpha=0.7)
plt.title('Tren TMA Seiring Waktu (Sampel 3 Stasiun)')
plt.legend()
plt.show()'''),

    new_markdown_cell('## 4. Data Preprocessing & Feature Engineering\n\nKita akan mengekstrak `datetime` dan `nama_pos` dari data test, kemudian menggabungkan data lingkungan (agregasi 6 jam) dan melakukan ekstraksi lag features secara optimal.'),
    new_code_cell('''# Ekstrak datetime dan nama_pos dari kolom id pada test set
def extract_test_info(df):
    df_copy = df.copy()
    # Format id: '2025-09-19 06:00:00 - Arjowinangun - Pacitan'
    split_info = df_copy['id'].str.split(' - ', n=1, expand=True)
    df_copy['datetime'] = pd.to_datetime(split_info[0])
    df_copy['nama_pos'] = split_info[1]
    return df_copy

test_processed = extract_test_info(test_df)
test_processed['is_test'] = 1
train_df['is_test'] = 0

# Gabungkan train dan test untuk feature engineering
full_df = pd.concat([train_df, test_processed], axis=0).reset_index(drop=True)

print(\"Full data shape:\", full_df.shape)'''),

    new_code_cell('''# Feature Engineering: Data Lingkungan
# Data lingkungan resolusinya tiap jam. Kita perlu agregasi tiap 6 jam (T-5 sampai T) ke waktu target (06:00, 12:00, 18:00)

lingkungan_df = lingkungan_df.sort_values(['nama_pos', 'datetime']).reset_index(drop=True)

# Memilih fitur lingkungan utama
num_cols = ['rainfall_mm', 'humidity_pct', 'temperature_c', 'wind_speed_kmh', 
            'soil_moisture_0_7cm', 'surface_pressure_hpa', 'solar_radiation_mj_m2']

def aggregate_environmental_features(env_df, cols_to_agg, window=6):
    agg_df = env_df[['datetime', 'nama_pos']].copy()
    grouped = env_df.groupby('nama_pos')
    
    for col in cols_to_agg:
        agg_df[f'{col}_mean_6h'] = grouped[col].transform(lambda x: x.rolling(window, min_periods=1).mean())
        agg_df[f'{col}_max_6h'] = grouped[col].transform(lambda x: x.rolling(window, min_periods=1).max())
        agg_df[f'{col}_sum_6h'] = grouped[col].transform(lambda x: x.rolling(window, min_periods=1).sum())
        
    return agg_df

print(\"Mengagregasi data lingkungan...\")
env_agg = aggregate_environmental_features(lingkungan_df, num_cols, window=6)

# Filter hanya pada jam pencatatan TMA (06:00, 12:00, 18:00)
env_agg = env_agg[env_agg['datetime'].dt.hour.isin([6, 12, 18])].reset_index(drop=True)

# Merge dengan full dataset
full_df = pd.merge(full_df, env_agg, on=['datetime', 'nama_pos'], how='left')

# Merge data koordinat stasiun
full_df = pd.merge(full_df, koordinat_df, on='nama_pos', how='left')
'''),
    
    new_code_cell('''# Feature Engineering: Time Features & Lags
full_df['hour'] = full_df['datetime'].dt.hour
full_df['day'] = full_df['datetime'].dt.day
full_df['month'] = full_df['datetime'].dt.month
full_df['dayofweek'] = full_df['datetime'].dt.dayofweek
full_df['is_weekend'] = full_df['dayofweek'].isin([5, 6]).astype(int)

# Fitur trigonometrik untuk month dan hour agar model menangkap cyclicity
full_df['hour_sin'] = np.sin(2 * np.pi * full_df['hour']/24.0)
full_df['hour_cos'] = np.cos(2 * np.pi * full_df['hour']/24.0)
full_df['month_sin'] = np.sin(2 * np.pi * full_df['month']/12.0)
full_df['month_cos'] = np.cos(2 * np.pi * full_df['month']/12.0)

# Sort untuk lag features
full_df = full_df.sort_values(['nama_pos', 'datetime']).reset_index(drop=True)

print(\"Membuat Lag features (Exogenous)...\")
lag_cols = [c for c in full_df.columns if 'mean_6h' in c or 'sum_6h' in c]
grouped_full = full_df.groupby('nama_pos')

for col in lag_cols:
    # Mengambil lag fitur cuaca dari 1, 2, dan 4 pencatatan sebelumnya (6, 12, dan 24 jam)
    full_df[f'{col}_lag1'] = grouped_full[col].shift(1)
    full_df[f'{col}_lag2'] = grouped_full[col].shift(2)
    full_df[f'{col}_lag4'] = grouped_full[col].shift(4)

# Label Encoding target / categorical
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
full_df['nama_pos_encoded'] = le.fit_transform(full_df['nama_pos'])

print(\"Feature Engineering Selesai. Total Kolom:\", len(full_df.columns))'''),

    new_markdown_cell('## 5. Model Selection & Training (XGBoost)'),
    new_code_cell('''# Pisahkan kembali train dan test
train_data = full_df[full_df['is_test'] == 0].copy()
test_data = full_df[full_df['is_test'] == 1].copy()

# Buang target missing di train (jika ada)
train_data = train_data.dropna(subset=['tma_mdpl']).reset_index(drop=True)

# Drop kolom identifier dan unneeded
drop_cols = ['datetime', 'nama_pos', 'tma_mdpl', 'id', 'is_test']
features = [c for c in train_data.columns if c not in drop_cols]

X = train_data[features]
y = train_data['tma_mdpl']

X_test = test_data[features]

print(f\"X_train shape: {X.shape}\")
print(f\"X_test shape: {X_test.shape}\")'''),

    new_code_cell('''# Setup K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
models = []
scores = []
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

# XGBoost hyperparameters optimal untuk tabular
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'learning_rate': 0.03,
    'max_depth': 7,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist', # Ganti ke 'gpu_hist' di Colab jika pakai GPU
    'random_state': 42
}

print(\"Memulai Training...\")
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f\"--- Fold {fold+1} ---\")
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_va, label=y_va)
    
    model = xgb.train(
        params, 
        dtrain, 
        num_boost_round=1500, 
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=100
    )
    
    val_preds = model.predict(dval)
    oof_preds[val_idx] = val_preds
    fold_rmse = root_mean_squared_error(y_va, val_preds)
    scores.append(fold_rmse)
    
    # Predict on test
    dtest = xgb.DMatrix(X_test)
    test_preds += model.predict(dtest) / kf.n_splits
    
    models.append(model)
    print(f\"Fold {fold+1} RMSE: {fold_rmse:.4f}\\n\")

print(f\"Average CV RMSE: {np.mean(scores):.4f}\")'''),

    new_markdown_cell('## 6. Model Evaluation & Feature Importance'),
    new_code_cell('''# Plot Feature Importance dari fold pertama
plt.figure(figsize=(10, 8))
xgb.plot_importance(models[0], max_num_features=20, height=0.5)
plt.title('Top 20 Feature Importance (XGBoost)')
plt.show()'''),

    new_markdown_cell('## 7. Generate File Submission\n\nMenghasilkan output `submission_xgboost_tma.csv` yang sama persis formatnya dengan `sample_submission.csv`.'),
    new_code_cell('''# Pasangkan prediksi dengan data test
test_data['tma_mdpl'] = test_preds

# Pastikan id berformat sama dengan awal
sub = test_data[['id', 'tma_mdpl']].copy()

# Sortir berdasarkan id atau kembalikan ke urutan asli di sample_submission.csv
sub = pd.merge(sample_sub[['id']], sub, on='id', how='left')

# Tampilkan 10 baris pertama untuk memastikan bentuknya
print(\"Sample Submission (First 10 Rows):\")
display(sub.head(10))

# Simpan ke CSV
sub.to_csv('submission_xgboost_tma.csv', index=False)
print(\"\\n✅ File 'submission_xgboost_tma.csv' berhasil disimpan!\")
''')
])

with open('XGBoost_Baseline_TMA.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
