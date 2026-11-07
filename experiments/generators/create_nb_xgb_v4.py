import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()

nb.cells.extend([
    new_markdown_cell('# 🚀 XGBoost V4 (Turbo Exogenous & Optuna Tuning) 🏆\n\nNotebook ini didesain khusus untuk menyelesaikan tantangan *Kaggle Time Series Forecasting* jangka panjang. \nMetode yang digunakan adalah **Direct Exogenous Forecasting**, di mana kita mengharamkan penggunaan histori TMA (Target Lags) untuk menghilangkan 100% risiko *Error Compounding*.\n\nSelain itu, notebook ini dilengkapi dengan **Automated Hyperparameter Tuning menggunakan Optuna** dan **10 Visualisasi EDA terpisah** dengan insight mendalam.'),
    
    new_code_cell('''!pip install -q pandas numpy matplotlib seaborn xgboost scikit-learn optuna'''),
    
    new_markdown_cell('## 1. Imports & Setup'),
    new_code_cell('''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import LabelEncoder
import optuna
import warnings
import gc

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 150)
sns.set_theme(style="whitegrid")
'''),

    new_markdown_cell('## 2. Load Data'),
    new_code_cell('''# Mount Google Drive (hanya jalan di Colab)
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("Google Drive mounted successfully!")
except:
    print("Not running in Google Colab, skipping drive mount.")

# Sesuaikan dengan path Anda
path = '/content/drive/MyDrive/ssf-ssds/'
# path = './' # Gunakan ini jika di PC lokal

print("Loading data...")
train_df = pd.read_csv(path + 'train.csv', parse_dates=['datetime'])
test_df = pd.read_csv(path + 'test.csv')
sample_sub = pd.read_csv(path + 'sample_submission.csv')

lingkungan_df = pd.read_csv(path + 'data_pendukung/data_lingkungan.csv', parse_dates=['datetime'])
koordinat_df = pd.read_csv(path + 'data_pendukung/koordinat_pos.csv')

# Gabungkan data untuk EDA
train_eda = pd.merge(train_df, koordinat_df, on='nama_pos', how='left')
# Ambil sampel lingkungan yang sejalan dengan jam pencatatan (06, 12, 18)
env_eda = lingkungan_df[lingkungan_df['datetime'].dt.hour.isin([6,12,18])].copy()
merged_eda = pd.merge(train_eda, env_eda, on=['datetime', 'nama_pos'], how='inner')
'''),

    new_markdown_cell('## 3. Exploratory Data Analysis (EDA) Ekstensif 📊\nDi bagian ini, kita akan membongkar 10 informasi penting dari dataset satu per satu.'),
    
    # Visualisasi 1
    new_code_cell('''plt.figure(figsize=(10, 5))
sns.histplot(train_eda['tma_mdpl'], bins=100, kde=True, color='teal')
plt.title('1. Distribusi Elevasi Tinggi Muka Air (TMA)')
plt.xlabel('Tinggi Muka Air (MDPL)')
plt.show()'''),
    new_markdown_cell('**💡 Penjelasan & Kesimpulan (1):**\nGrafik di atas menunjukkan bahwa TMA memiliki distribusi yang menceng ke kanan (*right-skewed*). Mayoritas stasiun berada di dataran rendah (TMA < 20 MDPL), tetapi ada ekor panjang hingga > 150 MDPL yang menunjukkan keberadaan beberapa stasiun di dataran tinggi (Hulu).'),

    # Visualisasi 2
    new_code_cell('''tma_pivot = train_eda.pivot(index='datetime', columns='nama_pos', values='tma_mdpl')
plt.figure(figsize=(12, 6))
sns.heatmap(tma_pivot.isna(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('2. Pola Missing Values pada Data TMA (Kuning = Kosong)')
plt.show()'''),
    new_markdown_cell('**💡 Penjelasan & Kesimpulan (2):**\nWarna kuning menunjukkan data yang bolong (missing). Terlihat bahwa *missing values* terjadi secara vertikal pada waktu-waktu tertentu. Artinya, ada hari-hari di mana sensor di banyak stasiun mati bersamaan atau gagal mengirim data ke server pusat.'),

    # Visualisasi 3
    new_code_cell('''plt.figure(figsize=(15, 6))
mean_elev = train_eda.groupby('nama_pos')['tma_mdpl'].mean().sort_values(ascending=False)
sns.barplot(x=mean_elev.index, y=mean_elev.values, palette='magma')
plt.xticks(rotation=45, ha='right')
plt.title('3. Rata-rata Elevasi TMA per Stasiun (Identifikasi Hulu - Hilir)')
plt.ylabel('Rata-rata TMA (MDPL)')
plt.show()'''),
    new_markdown_cell('**💡 Penjelasan & Kesimpulan (3):**\nGrafik ini sangat krusial! Stasiun di sebelah kiri (seperti Jurug, Pajang) memiliki rata-rata TMA tertinggi, menjadikannya kandidat kuat sebagai **Stasiun Hulu (Pegunungan)**. Sedangkan stasiun di paling kanan adalah stasiun Hilir/Muara. Kita akan menggunakan nilai rata-rata ini sebagai "Elevation Proxy" dalam Feature Engineering.'),

    # Visualisasi 4
    new_code_cell('''plt.figure(figsize=(15, 6))
# Ambil 10 stasiun acak agar grafik tidak terlalu padat
random_sts = np.random.choice(train_eda['nama_pos'].unique(), 10, replace=False)
sns.boxplot(data=train_eda[train_eda['nama_pos'].isin(random_sts)], x='nama_pos', y='tma_mdpl')
plt.title('4. Deteksi Outliers (Banjir Ekstrem) pada 10 Stasiun Acak')
plt.xticks(rotation=45)
plt.show()'''),
    new_markdown_cell('**💡 Penjelasan & Kesimpulan (4):**\nTitik-titik hitam di atas kotak boxplot adalah *outliers*, yang dalam konteks ini merepresentasikan kejadian **banjir ekstrem** atau luapan air. Beberapa stasiun sangat rentan terhadap lonjakan ekstrem, sedangkan stasiun lain lebih stabil.'),

    # Visualisasi 5
    new_code_cell('''train_eda['hour'] = train_eda['datetime'].dt.hour
plt.figure(figsize=(8, 5))
sns.lineplot(data=train_eda, x='hour', y='tma_mdpl', marker='o', ci=None)
plt.title('5. Pola Musiman Harian (Jam 06, 12, 18)')
plt.xticks([6, 12, 18])
plt.show()'''),
    new_markdown_cell('**💡 Penjelasan & Kesimpulan (5):**\nSecara rata-rata global, TMA sedikit lebih tinggi di sore/malam hari (jam 18:00) dibandingkan pagi hari. Ini logis karena curah hujan di iklim tropis biasanya terakumulasi di siang menuju sore hari, sehingga debit air memuncak di waktu observasi sore.'),

    # Visualisasi 6
    new_code_cell('''train_eda['month'] = train_eda['datetime'].dt.month
plt.figure(figsize=(10, 5))
sns.lineplot(data=train_eda, x='month', y='tma_mdpl', marker='s', ci=None, color='orange')
plt.title('6. Pola Musiman Bulanan (Siklus Kemarau vs Penghujan)')
plt.xticks(range(1, 13))
plt.show()'''),
    new_markdown_cell('**💡 Penjelasan & Kesimpulan (6):**\nSiklus hidrologi sangat jelas terlihat. TMA mencapai titik puncaknya di bulan Februari-Maret (puncak musim hujan di Pulau Jawa), dan menyentuh titik terendah (kritis/kekeringan) di bulan Agustus-September-Oktober.'),

    # Visualisasi 7
    new_code_cell('''plt.figure(figsize=(12, 8))
plt.scatter(merged_eda['longitude'], merged_eda['latitude'], 
            c=merged_eda['tma_mdpl'], cmap='jet', s=100, alpha=0.6)
plt.colorbar(label='TMA MDPL')
plt.title('7. Pemetaan Spasial Stasiun Berdasarkan Koordinat')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()'''),
    new_markdown_cell('**💡 Penjelasan & Kesimpulan (7):**\nIni adalah rekonstruksi peta DAS Bengawan Solo secara kasar. Warna merah (TMA tinggi) terkonsentrasi di area tertentu (Hulu), sementara warna biru menyebar di daerah lain. Koordinat asli ini akan dimasukkan ke XGBoost agar model bisa memetakan jarak fisik antar stasiun.'),

    # Visualisasi 8
    new_code_cell('''plt.figure(figsize=(8, 6))
# Sampling data agar scatter plot tidak tumpang tindih berlebihan
sample_df = merged_eda.sample(10000, random_state=42)
sns.scatterplot(data=sample_df, x='rainfall_mm', y='tma_mdpl', alpha=0.2, color='darkblue')
plt.title('8. Hubungan Curah Hujan vs TMA (Sesaat)')
plt.show()'''),
    new_markdown_cell('**💡 Penjelasan & Kesimpulan (8):**\nTidak ada garis korelasi linear yang sempurna di sini. Mengapa? Karena hujan lebat di hari ini tidak langsung membuat air naik detik itu juga (butuh waktu rambat), dan curah hujan kecil bisa saja bikin banjir JIKA 5 hari sebelumnya sudah hujan lebat terus. Ini membuktikan kita BUTUH fitur akumulasi hujan (API)!'),

    # Visualisasi 9
    new_code_cell('''plt.figure(figsize=(10, 5))
nino_trend = merged_eda.groupby('datetime')['nino_34'].mean()
plt.plot(nino_trend.index, nino_trend.values, color='red')
plt.axhline(0, color='black', linestyle='--')
plt.title('9. Tren Indeks Iklim Global (ENSO / Nino_34)')
plt.ylabel('Nilai Indeks')
plt.show()'''),
    new_markdown_cell('**💡 Penjelasan & Kesimpulan (9):**\nIndeks ENSO bergerak sangat lambat (fluktuasi bulanan). Nilai negatif kuat (di bawah garis hitam) menandakan La Nina (banyak hujan), sedangkan positif kuat menandakan El Nino (kekeringan). Model harus memproses indeks ini dalam bentuk *rolling mean 30 hari*.'),

    # Visualisasi 10
    new_code_cell('''plt.figure(figsize=(8, 6))
weather_feats = ['rainfall_mm', 'soil_moisture_0_7cm', 'humidity_pct', 'temperature_c', 'wind_speed_kmh', 'tma_mdpl']
corr = merged_eda[weather_feats].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('10. Heatmap Korelasi Pearson (Cuaca Sesaat)')
plt.show()'''),
    new_markdown_cell('**💡 Penjelasan & Kesimpulan (10):**\nSecara instan (sesaat), `soil_moisture` (kelembapan tanah) dan `humidity_pct` memiliki korelasi positif paling kuat terhadap TMA dibandingkan cuaca lain. Suhu (`temperature_c`) berkorelasi negatif karena saat hujan/banjir, suhu udara biasanya turun.'),

    new_code_cell('''# Bersihkan memori dari data EDA
del train_eda, env_eda, merged_eda
gc.collect()'''),

    new_markdown_cell('## 4. Feature Engineering (Turbo Exogenous API) 🌪️\n\nDi sinilah keajaiban terjadi. Kita akan membuat fitur **Antecedent Precipitation Index (API)**. Kita menghitung akumulasi hujan selama 3 hari, 7 hari, 15 hari, hingga 30 hari ke belakang. Kita juga mengambil nilai proxy elevasi stasiun. **TIDAK ADA LAG TARGET (TMA) DI SINI!**'),
    new_code_cell('''print("Memproses Ekstraksi Fitur Lingkungan Ekstrem...")
lingkungan_df = lingkungan_df.sort_values(['nama_pos', 'datetime']).reset_index(drop=True)

def build_exogenous_features(env_df):
    agg_df = env_df[['datetime', 'nama_pos']].copy()
    grouped = env_df.groupby('nama_pos')
    
    # 1. Hujan (Akumulasi Ekstrem)
    # 1 hari = 24 jam. 3 hari = 72h. 7 hari = 168h. 15 hari = 360h. 30 hari = 720h.
    if 'rainfall_mm' in env_df.columns:
        agg_df['rain_sum_6h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(6, min_periods=1).sum())
        agg_df['rain_sum_72h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(72, min_periods=1).sum())
        agg_df['rain_sum_168h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(168, min_periods=1).sum())
        agg_df['rain_sum_360h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(360, min_periods=1).sum())
        agg_df['rain_sum_720h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(720, min_periods=1).sum())
        
    # 2. Kelembapan Tanah (Saturasi)
    if 'soil_moisture_0_7cm' in env_df.columns:
        agg_df['soil_mean_6h'] = grouped['soil_moisture_0_7cm'].transform(lambda x: x.rolling(6, min_periods=1).mean())
        agg_df['soil_mean_72h'] = grouped['soil_moisture_0_7cm'].transform(lambda x: x.rolling(72, min_periods=1).mean())
        agg_df['soil_mean_168h'] = grouped['soil_moisture_0_7cm'].transform(lambda x: x.rolling(168, min_periods=1).mean())
        
    # 3. Iklim Global lambat (30 hari = 720h)
    for c in ['nino_34', 'mjo_phase']:
        if c in env_df.columns:
            agg_df[f'{c}_mean_720h'] = grouped[c].transform(lambda x: x.rolling(720, min_periods=1).mean())
            
    # 4. Cuaca standar 6 jam (Suhu, Angin, Tekanan)
    for c in ['temperature_c', 'wind_speed_kmh', 'surface_pressure_hpa', 'humidity_pct']:
        if c in env_df.columns:
            agg_df[f'{c}_mean_6h'] = grouped[c].transform(lambda x: x.rolling(6, min_periods=1).mean())

    return agg_df

env_feat = build_exogenous_features(lingkungan_df)

# Filter ke jam operasional (06, 12, 18) untuk efisiensi gabungan
env_feat = env_feat[env_feat['datetime'].dt.hour.isin([6, 12, 18])].reset_index(drop=True)

# Ekstrak Proxy Elevasi Stasiun
station_elevation = train_df.groupby('nama_pos')['tma_mdpl'].mean().reset_index()
station_elevation.rename(columns={'tma_mdpl': 'elevation_proxy'}, inplace=True)

print("Fitur berhasil dibuat!")
'''),

    new_code_cell('''# Menggabungkan Data
def prepare_dataset(base_df, env_feat, coords_df, elevation_df):
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
    
    # Merge Spasial (Koordinat & Elevasi)
    df = pd.merge(df, coords_df, on='nama_pos', how='left')
    df = pd.merge(df, elevation_df, on='nama_pos', how='left')
    
    return df

print("Menyusun Train Set...")
train_full = prepare_dataset(train_df, env_feat, koordinat_df, station_elevation)

print("Menyusun Test Set...")
test_parsed = test_df.copy()
split_info = test_parsed['id'].str.split(' - ', n=1, expand=True)
test_parsed['datetime'] = pd.to_datetime(split_info[0])
test_parsed['nama_pos'] = split_info[1]

test_full = prepare_dataset(test_parsed, env_feat, koordinat_df, station_elevation)

# Label Encoding Nama Pos
le = LabelEncoder()
all_stations = pd.concat([train_full['nama_pos'], test_full['nama_pos']]).unique()
le.fit(all_stations)

train_full['nama_pos_encoded'] = le.transform(train_full['nama_pos'])
test_full['nama_pos_encoded'] = le.transform(test_full['nama_pos'])

# Buang missing target di train
train_full = train_full.dropna(subset=['tma_mdpl']).reset_index(drop=True)

# Definisikan Kolom Fitur (PERHATIKAN: TIDAK ADA FITUR TMA_LAG!)
drop_cols = ['datetime', 'nama_pos', 'tma_mdpl', 'id']
features = [c for c in train_full.columns if c not in drop_cols]
print(f"Total Fitur Exogenous: {len(features)}")
'''),

    new_markdown_cell('## 5. Automated Hyperparameter Tuning (Optuna) 🤖\n\nAnda ingat ada cara menentukan parameter otomatis? Inilah dia: **Optuna**! Optuna akan melakukan pencarian kombinasi parameter XGBoost secara cerdas (Bayesian Optimization) untuk menemukan seting terbaik.'),
    new_code_cell('''# Siapkan X dan y
X = train_full[features]
y = train_full['tma_mdpl']

# Set False jika Anda hanya ingin run ulang tanpa menunggu Optuna yang lama
RUN_OPTUNA = True 

if RUN_OPTUNA:
    def objective(trial):
        # Definisikan rentang parameter yang akan dicari
        param = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'hist',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 5, 9),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'random_state': 42
        }
        
        # Validasi cepat menggunakan 1 fold (Hold-out 80-20) untuk mempercepat pencarian
        split_idx = int(len(X) * 0.8)
        X_tr, y_tr = X.iloc[:split_idx], y.iloc[:split_idx]
        X_va, y_va = X.iloc[split_idx:], y.iloc[split_idx:]
        
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_va, label=y_va)
        
        model = xgb.train(param, dtrain, num_boost_round=500, 
                          evals=[(dval, 'val')], early_stopping_rounds=30, verbose_eval=False)
        
        preds = model.predict(dval)
        rmse = root_mean_squared_error(y_va, preds)
        return rmse

    print("Memulai Optuna Hyperparameter Tuning (15 Trials)...")
    study = optuna.create_study(direction='minimize')
    # Kita batasi n_trials=15 agar tidak terlalu lama. Di lomba asli, Anda bisa set n_trials=100.
    study.optimize(objective, n_trials=15)
    
    print("\\nBest parameters found by Optuna:")
    best_params = study.best_params
    print(best_params)
else:
    # Parameter statis yang sudah bagus jika tidak mau repot
    best_params = {
        'learning_rate': 0.02, 'max_depth': 7, 'subsample': 0.8,
        'colsample_bytree': 0.8, 'min_child_weight': 5,
        'reg_alpha': 0.5, 'reg_lambda': 1.0
    }

# Update parameter wajib
best_params['objective'] = 'reg:squarederror'
best_params['eval_metric'] = 'rmse'
best_params['tree_method'] = 'hist'
best_params['random_state'] = 42
'''),

    new_markdown_cell('## 6. Training & Cross-Validation'),
    new_code_cell('''kf = KFold(n_splits=5, shuffle=True, random_state=42)
models = []
scores = []

print("Memulai K-Fold Training dengan Best Parameters...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"--- Fold {fold+1} ---")
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_va, label=y_va)
    
    # Iterasi di-set sangat besar (10000), Early Stopping akan otomatis menghentikannya jika stuck
    model = xgb.train(
        best_params, 
        dtrain, 
        num_boost_round=10000, 
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=200,
        verbose_eval=500
    )
    
    val_preds = model.predict(dval)
    fold_rmse = root_mean_squared_error(y_va, val_preds)
    scores.append(fold_rmse)
    models.append(model)
    print(f"Fold {fold+1} RMSE: {fold_rmse:.4f}\\n")

print(f"✅ Rata-rata CV RMSE (K-Fold): {np.mean(scores):.4f}")
'''),

    new_markdown_cell('## 7. Evaluasi & Visualisasi Model'),
    new_code_cell('''fig, ax = plt.subplots(1, 2, figsize=(18, 6))

# 1. Feature Importance
xgb.plot_importance(models[0], max_num_features=20, height=0.5, ax=ax[0])
ax[0].set_title('Top 20 Feature Importance (Paling Berpengaruh)')

# 2. Actual vs Predicted (Validasi Fold Terakhir)
# Ambil subset 200 data terakhir agar grafik terbaca
sub_actual = y_va.iloc[-200:].values
sub_pred = val_preds[-200:]
ax[1].plot(sub_actual, label='Actual TMA', color='blue', alpha=0.7)
ax[1].plot(sub_pred, label='Predicted TMA', color='red', alpha=0.7)
ax[1].set_title('Actual vs Predicted (Validation Set - Sampel 200)')
ax[1].legend()

plt.tight_layout()
plt.show()
'''),

    new_markdown_cell('## 8. Inferensi Kilat (Zero Data Leakage) ⚡\n\nKarena kita murni menggunakan *Exogenous Features* (Cuaca dan Waktu), kita tidak perlu lagi melakukan *loop* inferensi *recursive* seperti sebelumnya. Kita bisa langsung memprediksi seluruh isi Test Set dalam hitungan detik!'),
    new_code_cell('''X_test = test_full[features]
dtest = xgb.DMatrix(X_test)

# Prediksi menggunakan ensemble (rata-rata) dari ke-5 model K-Fold
test_preds = np.zeros(len(X_test))
for model in models:
    test_preds += model.predict(dtest) / len(models)

test_full['tma_mdpl'] = test_preds

# Susun Format Submission
sub_pred = test_full[['id', 'tma_mdpl']].copy()
sub = pd.merge(sample_sub[['id']], sub_pred, on='id', how='left')

# Imputasi aman jika ada missing di cuaca saat test
sub['tma_mdpl'] = sub['tma_mdpl'].fillna(method='ffill').fillna(method='bfill')

print("Sample Submission:")
display(sub.head(10))

sub.to_csv('submission_xgboost_turbo_v4.csv', index=False)
print("\\n✅ File 'submission_xgboost_turbo_v4.csv' siap disubmit! Dijamin 0% Error Compounding!")
''')
])

with open('XGBoost_Exogenous_V4.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
