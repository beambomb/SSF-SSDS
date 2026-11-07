import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()

nb.cells.extend([
    new_markdown_cell('# 🌪️ V6 FINAL: The Ultimate Exogenous Features & Holy Trinity\n\nNotebook ini adalah versi penyempurnaan (Final Form) yang mencakup **10 EDA Lengkap**, **Visualisasi Evaluasi**, dan **Fitur Hidrologi Tingkat Dewa** (Runoff Proxy, Soil Gradient, MJO Kinematics, Wind Cyclical, dan Landcover) sambil tetap mempertahankan arsitektur Ensemble 3 Model (XGB + LGBM + Cat).'),
    
    new_code_cell('''!pip install -q pandas numpy matplotlib seaborn xgboost catboost lightgbm scikit-learn optuna'''),
    
    new_markdown_cell('## 1. Imports & Setup'),
    new_code_cell('''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

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
    new_markdown_cell('**💡 Penjelasan & Kesimpulan (8):**\nTidak ada garis korelasi linear yang sempurna di sini. Curah hujan kecil bisa saja bikin banjir JIKA 5 hari sebelumnya sudah hujan lebat terus. Ini membuktikan kita BUTUH fitur akumulasi hujan (API)!'),

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

    new_markdown_cell('## 4. Ekstraksi Fitur (The Ultimate Hydrological Features) 💧🌪️\n\nKita menerapkan saran riset *State-of-the-Art*:\n1. **Runoff Proxy:** `rainfall * soil_moisture`\n2. **Soil Gradient:** Daya serap tanah per jam\n3. **MJO Speed:** Kinematika iklim global `sqrt(rmm1^2 + rmm2^2)`\n4. **Cyclical Wind:** `sin` dan `cos` dari arah angin\n5. **Landcover & Built Surface:** Dimasukkan sebagai fitur spasial.'),
    new_code_cell('''print("Memproses Ekstraksi Fitur Ultimate...")
lingkungan_df = lingkungan_df.sort_values(['nama_pos', 'datetime']).reset_index(drop=True)

def build_ultimate_features(env_df):
    agg_df = env_df[['datetime', 'nama_pos']].copy()
    grouped = env_df.groupby('nama_pos')
    
    # --- 1. RAINFALL LAGS (Hourly) ---
    if 'rainfall_mm' in env_df.columns:
        agg_df['rain_lag_1h'] = grouped['rainfall_mm'].shift(1)
        agg_df['rain_lag_2h'] = grouped['rainfall_mm'].shift(2)
        agg_df['rain_lag_3h'] = grouped['rainfall_mm'].shift(3)
        agg_df['rain_lag_6h'] = grouped['rainfall_mm'].shift(6)
        agg_df['rain_lag_12h'] = grouped['rainfall_mm'].shift(12)
        
        # Accumulations
        agg_df['rain_sum_6h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(6, min_periods=1).sum())
        agg_df['rain_sum_72h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(72, min_periods=1).sum())
        agg_df['rain_sum_168h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(168, min_periods=1).sum())
        agg_df['rain_sum_720h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(720, min_periods=1).sum())

    # --- 2. SOIL & RUNOFF PROXY ---
    if 'soil_moisture_0_7cm' in env_df.columns:
        # Gradient daya serap tanah
        agg_df['soil_grad_1h'] = grouped['soil_moisture_0_7cm'].diff()
        
        # Runoff Proxy (Interaksi non-linear)
        agg_df['runoff_proxy'] = env_df['rainfall_mm'] * env_df['soil_moisture_0_7cm']
        
        agg_df['soil_mean_6h'] = grouped['soil_moisture_0_7cm'].transform(lambda x: x.rolling(6, min_periods=1).mean())
        agg_df['soil_mean_168h'] = grouped['soil_moisture_0_7cm'].transform(lambda x: x.rolling(168, min_periods=1).mean())

    # --- 3. WIND CYCLICAL TRANSFORM ---
    if 'wind_direction_deg' in env_df.columns:
        agg_df['wind_sin'] = np.sin(2 * np.pi * env_df['wind_direction_deg'] / 360.0)
        agg_df['wind_cos'] = np.cos(2 * np.pi * env_df['wind_direction_deg'] / 360.0)

    # --- 4. MJO KINEMATICS & MACRO CLIMATE ---
    if 'rmm1' in env_df.columns and 'rmm2' in env_df.columns:
        agg_df['mjo_speed'] = np.sqrt(env_df['rmm1']**2 + env_df['rmm2']**2)
        agg_df['mjo_speed_mean_720h'] = agg_df.groupby('nama_pos')['mjo_speed'].transform(lambda x: x.rolling(720, min_periods=1).mean())
        
    for c in ['nino_34', 'mjo_phase']:
        if c in env_df.columns:
            agg_df[f'{c}_mean_720h'] = grouped[c].transform(lambda x: x.rolling(720, min_periods=1).mean())

    # --- 5. SPATIAL LANDCOVER ---
    if 'built_surface_m2' in env_df.columns:
        agg_df['built_surface_m2'] = env_df['built_surface_m2']
    if 'landcover_name' in env_df.columns:
        agg_df['landcover_name'] = env_df['landcover_name']
        
    # Other standard weather
    for c in ['temperature_c', 'surface_pressure_hpa', 'humidity_pct']:
        if c in env_df.columns:
            agg_df[f'{c}_mean_6h'] = grouped[c].transform(lambda x: x.rolling(6, min_periods=1).mean())

    return agg_df

env_feat = build_ultimate_features(lingkungan_df)
env_feat = env_feat[env_feat['datetime'].dt.hour.isin([6, 12, 18])].reset_index(drop=True)

station_elevation = train_df.groupby('nama_pos')['tma_mdpl'].mean().reset_index()
station_elevation.rename(columns={'tma_mdpl': 'elevation_proxy'}, inplace=True)
print("Seluruh fitur berhasil diekstrak!")
'''),

    new_code_cell('''def prepare_dataset(base_df, env_feat, coords_df, elevation_df):
    df = base_df.copy()
    
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['dayofyear'] = df['datetime'].dt.dayofyear
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24.0)
    
    df = pd.merge(df, env_feat, on=['datetime', 'nama_pos'], how='left')
    df = pd.merge(df, coords_df, on='nama_pos', how='left')
    df = pd.merge(df, elevation_df, on='nama_pos', how='left')
    return df

print("Menyusun Train & Test Set...")
train_full = prepare_dataset(train_df, env_feat, koordinat_df, station_elevation)

test_parsed = test_df.copy()
split_info = test_parsed['id'].str.split(' - ', n=1, expand=True)
test_parsed['datetime'] = pd.to_datetime(split_info[0])
test_parsed['nama_pos'] = split_info[1]
test_full = prepare_dataset(test_parsed, env_feat, koordinat_df, station_elevation)

le_pos = LabelEncoder()
all_stations = pd.concat([train_full['nama_pos'], test_full['nama_pos']]).unique()
train_full['nama_pos_encoded'] = le_pos.fit_transform(train_full['nama_pos'])
test_full['nama_pos_encoded'] = le_pos.transform(test_full['nama_pos'])

if 'landcover_name' in train_full.columns:
    le_land = LabelEncoder()
    train_full['landcover_name'] = train_full['landcover_name'].fillna('Unknown')
    test_full['landcover_name'] = test_full['landcover_name'].fillna('Unknown')
    train_full['landcover_encoded'] = le_land.fit_transform(train_full['landcover_name'])
    test_full['landcover_encoded'] = le_land.transform(test_full['landcover_name'])

train_full = train_full.dropna(subset=['tma_mdpl']).reset_index(drop=True)

drop_cols = ['datetime', 'nama_pos', 'tma_mdpl', 'id', 'landcover_name']
features = [c for c in train_full.columns if c not in drop_cols]
print(f"Total Fitur Ultimate: {len(features)}")
'''),

    new_markdown_cell('## 5. Log Transformation & Setup X, y'),
    new_code_cell('''X = train_full[features]
y_target = train_full['tma_mdpl']

print("Target siap digunakan (Tanpa Log Transform untuk menjaga akurasi puncak TMA)!")
'''),

    new_markdown_cell('## 6. Training "The Kaggle Holy Trinity" (K-Fold)\nMelatih XGBoost, LightGBM, dan CatBoost. Optuna dibatasi ke setingan andal untuk efisiensi.'),
    new_code_cell('''kf = KFold(n_splits=5, shuffle=True, random_state=42)

xgb_models = []
lgb_models = []
cat_models = []

xgb_params = {
    'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'tree_method': 'hist',
    'learning_rate': 0.03, 'max_depth': 8, 'subsample': 0.8, 'random_state': 42
}

lgb_params = {
    'objective': 'regression', 'metric': 'rmse', 'learning_rate': 0.03,
    'num_leaves': 64, 'max_depth': 8, 'feature_fraction': 0.8,
    'bagging_fraction': 0.8, 'bagging_freq': 1, 'verbose': -1, 'random_state': 42
}

cat_features_cols = ['nama_pos_encoded']
if 'landcover_encoded' in features:
    cat_features_cols.append('landcover_encoded')
lgb_cat_idx = [features.index(c) for c in cat_features_cols]
catboost_cat_idx = lgb_cat_idx.copy()

print("Mulai melatih TRINITY K-FOLD (XGB, LGBM, CAT)...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_target)):
    print(f"\\n========== FOLD {fold+1} ==========")
    X_tr, y_tr = X.iloc[train_idx], y_target.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y_target.iloc[val_idx]
    
    # 1. XGBoost
    dtrain_xgb = xgb.DMatrix(X_tr, label=y_tr)
    dval_xgb = xgb.DMatrix(X_va, label=y_va)
    model_xgb = xgb.train(xgb_params, dtrain_xgb, num_boost_round=3000, 
                          evals=[(dtrain_xgb, 'train'), (dval_xgb, 'val')],
                          early_stopping_rounds=50, verbose_eval=False) # disilent agar log bersih
    xgb_models.append(model_xgb)
    
    # 2. LightGBM
    dtrain_lgb = lgb.Dataset(X_tr, label=y_tr, categorical_feature=lgb_cat_idx)
    dval_lgb = lgb.Dataset(X_va, label=y_va, reference=dtrain_lgb, categorical_feature=lgb_cat_idx)
    model_lgb = lgb.train(lgb_params, dtrain_lgb, num_boost_round=3000,
                          valid_sets=[dtrain_lgb, dval_lgb],
                          callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
    lgb_models.append(model_lgb)
    
    # 3. CatBoost
    model_cat = CatBoostRegressor(iterations=3000, learning_rate=0.03, depth=8,
                                  eval_metric='RMSE', random_seed=42, verbose=False)
    model_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), 
                  cat_features=catboost_cat_idx, early_stopping_rounds=50)
    cat_models.append(model_cat)
    
    print(f"Fold {fold+1} Selesai dilatih!")
'''),

    new_markdown_cell('## 7. Evaluasi & Visualisasi Model 📈\nMari kita lihat fitur mana yang menjadi penentu dan seberapa akurat tebakan model Ensemble pada data Validation.'),
    new_code_cell('''fig, ax = plt.subplots(1, 2, figsize=(18, 6))

# 1. Feature Importance (XGBoost Fold 1)
xgb.plot_importance(xgb_models[0], max_num_features=20, height=0.5, ax=ax[0])
ax[0].set_title('Top 20 Feature Importance (XGBoost)')

# 2. Actual vs Predicted (Validasi Fold Terakhir)
# Ambil subset 200 data terakhir agar grafik terbaca
sub_actual = y_va.iloc[-200:].values

# Prediksi Ensemble di Validation Set fold terakhir
val_pred_xgb = xgb_models[-1].predict(dval_xgb)[-200:]
val_pred_lgb = lgb_models[-1].predict(X_va)[-200:]
val_pred_cat = cat_models[-1].predict(X_va)[-200:]

# Blend
sub_pred = (0.4 * val_pred_xgb) + (0.3 * val_pred_lgb) + (0.3 * val_pred_cat)

ax[1].plot(sub_actual, label='Actual TMA (Asli)', color='blue', alpha=0.7)
ax[1].plot(sub_pred, label='Predicted TMA (Blended)', color='red', alpha=0.7)
ax[1].set_title('Actual vs Predicted (Validation Set - Sampel 200)')
ax[1].legend()

plt.tight_layout()
plt.show()
'''),

    new_markdown_cell('## 8. Ensemble Inference (Blending) ⚔️'),
    new_code_cell('''X_test = test_full[features]
dtest_xgb = xgb.DMatrix(X_test)

preds_xgb = np.zeros(len(X_test))
preds_lgb = np.zeros(len(X_test))
preds_cat = np.zeros(len(X_test))

# Prediksi menggunakan ensemble tiap Fold
for i in range(5):
    preds_xgb += xgb_models[i].predict(dtest_xgb) / 5
    preds_lgb += lgb_models[i].predict(X_test) / 5
    preds_cat += cat_models[i].predict(X_test) / 5

# BLENDING THE TRINITY: 40% XGBoost, 30% LightGBM, 30% CatBoost
final_preds = (0.4 * preds_xgb) + (0.3 * preds_lgb) + (0.3 * preds_cat)

test_full['tma_mdpl'] = final_preds

# Format Submission
sub_pred = test_full[['id', 'tma_mdpl']].copy()
sub = pd.merge(sample_sub[['id']], sub_pred, on='id', how='left')
sub['tma_mdpl'] = sub['tma_mdpl'].fillna(method='ffill').fillna(method='bfill')

display(sub.head(10))
sub.to_csv('submission_v6_final.csv', index=False)
print("\\n✅ THE ULTIMATE V6 FINAL COMPLETED! File 'submission_v6_final.csv' siap di-submit!")
''')
])

with open('V6_Final_TMA.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
