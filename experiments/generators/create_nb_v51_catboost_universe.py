import nbformat as nbf

nb = nbf.v4.new_notebook()

nb.cells = [
    nbf.v4.new_markdown_cell("# 😻 V51: THE CATBOOST UNIVERSE (Categorical Boosting) 😻\\nAnda menyadari kata kunci **'Categorical Boosting'**! Ya, jika kita melatih 1 Model Global untuk membedakan karakteristik 30 stasiun, maka raja sejati untuk tugas ini bukanlah XGBoost, melainkan algoritma buatan Yandex Rusia: **CatBoost**.\\n\\nCatBoost memiliki mekanisme *Ordered Boosting* yang mencegah kebocoran data saat memproses kolom kategori (`nama_pos`)."),
    
    nbf.v4.new_code_cell("""# 1. MOUNT GOOGLE DRIVE & INSTALL LIBRARIES
!pip install -q catboost optuna ipywidgets haversine

from google.colab import drive
import os
import warnings
warnings.filterwarnings('ignore')

print("Meminta Izin Akses Google Drive...")
drive.mount('/content/drive')

# UBAH BARIS INI SESUAI DENGAN FOLDER ANDA
PROJECT_PATH = '/content/drive/MyDrive/SSF-SSDS/'
os.chdir(PROJECT_PATH)
print(f"Bekerja di direktori: {os.getcwd()}")"""),
    
    nbf.v4.new_code_cell("""# 2. IMPORT LIBRARY & LOAD DATA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor, Pool
import optuna
from haversine import haversine, Unit
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

print("Memuat Dataset Utama...")
train_df = pd.read_csv('train.csv', parse_dates=['datetime'])
test_df = pd.read_csv('test.csv')
env_df = pd.read_csv('data_pendukung/data_lingkungan.csv', parse_dates=['datetime'])
sample_sub = pd.read_csv('sample_submission.csv')
coords_df = pd.read_csv('data_pendukung/koordinat_pos.csv')

# Parsing Test Set
test_parsed = test_df.copy()
split_info = test_parsed['id'].str.split(' - ', n=1, expand=True)
test_parsed['datetime'] = pd.to_datetime(split_info[0])
test_parsed['nama_pos'] = split_info[1]

print("Data berhasil dimuat!")"""),

    nbf.v4.new_markdown_cell("### TAHAP 1: MATRIKS JARING HAVERSINE (IDW SPATIAL LAG)"),
    nbf.v4.new_code_cell("""print("Membangun Matriks Jarak (Distance Matrix)...")

stations = coords_df['nama_pos'].values
dist_matrix = pd.DataFrame(index=stations, columns=stations, dtype=float)

for i, row_i in coords_df.iterrows():
    for j, row_j in coords_df.iterrows():
        if i == j:
            dist_matrix.loc[row_i['nama_pos'], row_j['nama_pos']] = 0.0
        else:
            p1 = (row_i['latitude'], row_i['longitude'])
            p2 = (row_j['latitude'], row_j['longitude'])
            dist_matrix.loc[row_i['nama_pos'], row_j['nama_pos']] = haversine(p1, p2, unit=Unit.KILOMETERS)

elevations = train_df.groupby('nama_pos')['tma_mdpl'].mean().to_dict()
print("Matriks siap!")"""),

    nbf.v4.new_markdown_cell("### TAHAP 2: FISIKA ALAM & INVERSE DISTANCE WEIGHTING"),
    nbf.v4.new_code_cell("""print("Mengekstraksi Fisika Tanah Dalam & Cuaca Ekstrem...")
env_df = env_df.sort_values(['nama_pos', 'datetime']).reset_index(drop=True)
grouped = env_df.groupby('nama_pos')

# Fisika Hujan
env_df['rain_sum_72h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(72, min_periods=1).sum())
env_df['rain_sum_168h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(168, min_periods=1).sum())
env_df['rain_ewm_48h'] = grouped['rainfall_mm'].transform(lambda x: x.ewm(halflife=48, ignore_na=True).mean())

# Fisika Tanah Dalam
if 'soil_moisture_100_255cm' in env_df.columns:
    env_df['deep_soil_ewm_168h'] = grouped['soil_moisture_100_255cm'].transform(lambda x: x.ewm(halflife=168, ignore_na=True).mean())
    env_df['soil_0_7cm_sum_72h'] = grouped['soil_moisture_0_7cm'].transform(lambda x: x.rolling(72, min_periods=1).sum())
else:
    print("Warning: Data tanah dalam tidak ditemukan!")

print("Menghitung IDW Spatial Lag (Gelombang Badai Regional)...")
pivot_rain = env_df.pivot(index='datetime', columns='nama_pos', values='rain_sum_168h')
idw_features = []
epsilon = 1.0 

for pos in stations:
    weights = 1.0 / (dist_matrix.loc[pos] + epsilon)
    weights[pos] = 0.0
    weight_sum = weights.sum()
    idw_rain = (pivot_rain * weights).sum(axis=1) / weight_sum
    
    idw_df = pd.DataFrame({
        'datetime': pivot_rain.index,
        'nama_pos': pos,
        'spatial_idw_rain_168h': idw_rain.values
    })
    idw_features.append(idw_df)

spatial_idw_df = pd.concat(idw_features, ignore_index=True)
env_df = pd.merge(env_df, spatial_idw_df, on=['datetime', 'nama_pos'], how='left')
print("Selesai!")"""),

    nbf.v4.new_markdown_cell("### TAHAP 3: MERAKIT DATASET UNTUK CATBOOST"),
    nbf.v4.new_code_cell("""def prepare_catboost_dataset(base_df, env_feat, historical_tma_df):
    df = base_df.copy()
    
    # Fitur Waktu Siklik
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['dayofyear'] = df['datetime'].dt.dayofyear
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12.0)
    df['year'] = df['datetime'].dt.year
    df['year_month'] = df['year'].astype(str) + "_" + df['month'].astype(str)
    
    # Gabung Lingkungan
    keep_env = ['datetime', 'nama_pos', 'rain_sum_72h', 'rain_sum_168h', 'rain_ewm_48h', 
                'deep_soil_ewm_168h', 'soil_0_7cm_sum_72h', 'spatial_idw_rain_168h', 
                'temperature_c', 'built_surface_m2']
    df = pd.merge(df, env_feat[keep_env], on=['datetime', 'nama_pos'], how='left')
    
    # Memori Stasiun Sendiri
    historical_map = historical_tma_df.set_index(['nama_pos', 'datetime'])['tma_mdpl'].to_dict()
    df['tma_lag_365d'] = df.apply(lambda row: historical_map.get((row['nama_pos'], row['datetime'] - pd.DateOffset(years=1)), np.nan), axis=1)
    
    last_known_map = historical_tma_df.dropna(subset=['tma_mdpl']).sort_values('datetime').groupby('nama_pos').tail(1).set_index('nama_pos')['tma_mdpl'].to_dict()
    df['last_known_tma'] = df['nama_pos'].map(last_known_map)
    
    # CatBoost TIDAK perlu tipe 'category', cukup dibiarkan sebagai String/Object!
    df['nama_pos'] = df['nama_pos'].astype(str)
    
    return df

historical_tma = train_df[['datetime', 'nama_pos', 'tma_mdpl']].copy()

print("Memproses Train & Test Set...")
train_full = prepare_catboost_dataset(train_df, env_df, historical_tma)
test_full = prepare_catboost_dataset(test_parsed, env_df, historical_tma)

for st in train_full['nama_pos'].unique():
    mask_tr = train_full['nama_pos'] == st
    train_full.loc[mask_tr, 'tma_lag_365d'] = train_full.loc[mask_tr, 'tma_lag_365d'].fillna(train_full.loc[mask_tr, 'last_known_tma'])
    mask_te = test_full['nama_pos'] == st
    test_full.loc[mask_te, 'tma_lag_365d'] = test_full.loc[mask_te, 'tma_lag_365d'].fillna(test_full.loc[mask_te, 'last_known_tma'])

train_full = train_full.dropna(subset=['tma_mdpl']).reset_index(drop=True)

drop_cols = ['datetime', 'tma_mdpl', 'id', 'year', 'year_month']
features = [c for c in train_full.columns if c not in drop_cols]

# Mendefinisikan kolom kategori (Sangat penting untuk CatBoost)
cat_features = ['nama_pos'] 

print(f"Total Fitur CatBoost V51: {len(features)}")
print("Categorical Features:", cat_features)"""),

    nbf.v4.new_markdown_cell("### TAHAP 4: OPTUNA TWEEDIE CATBOOST (The King of Categorical)"),
    nbf.v4.new_code_cell("""print("Mempersiapkan Global Training dengan CatBoost...")

X_global = train_full[features]
y_global = train_full['tma_mdpl']
groups = train_full['year_month']

def objective(trial):
    variance_power = trial.suggest_float('variance_power', 1.2, 1.8)
    params = {
        'loss_function': f'Tweedie:variance_power={variance_power}',
        'eval_metric': 'RMSE',
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.1, log=True),
        'depth': trial.suggest_int('depth', 5, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 20.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'iterations': 500,
        'random_seed': 42,
        'verbose': False,
        'task_type': 'CPU' # Ubah ke 'GPU' jika Colab Anda diset GPU
    }
    
    gkf = GroupKFold(n_splits=4)
    train_idx, val_idx = next(gkf.split(X_global, y_global, groups=groups))
    
    X_tr, y_tr = X_global.iloc[train_idx], np.clip(y_global.iloc[train_idx], 0.0001, None)
    X_va, y_va = X_global.iloc[val_idx], np.clip(y_global.iloc[val_idx], 0.0001, None)
    
    train_pool = Pool(X_tr, label=y_tr, cat_features=cat_features)
    val_pool = Pool(X_va, label=y_va, cat_features=cat_features)
    
    model = CatBoostRegressor(**params)
    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=30)
    
    preds = model.predict(val_pool)
    return np.sqrt(mean_squared_error(y_va, preds))

print("Mencari Hyperparameter CatBoost Terbaik (15 Trials)...")
optuna.logging.set_verbosity(optuna.logging.ERROR)
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=15)

best_params = study.best_params
opt_power = best_params.pop('variance_power')
best_params['loss_function'] = f'Tweedie:variance_power={opt_power}'
best_params['eval_metric'] = 'RMSE'
best_params['iterations'] = 800
best_params['random_seed'] = 42

print(f"\\n🏆 Best Single-Fold RMSE: {study.best_value:.4f}")
print("Melatih Model CatBoost Final dengan Seluruh Data Train...")
final_pool = Pool(X_global, label=np.clip(y_global, 0.0001, None), cat_features=cat_features)

model_final = CatBoostRegressor(**best_params)
model_final.fit(final_pool, verbose=100)
print("Selesai!")"""),

    nbf.v4.new_markdown_cell("### TAHAP 5: FEATURE IMPORTANCE & SUBMISSION"),
    nbf.v4.new_code_cell("""fig, ax = plt.subplots(figsize=(12, 8))
imp_df = pd.DataFrame({
    'Features': features, 
    'Importance score': model_final.get_feature_importance()
}).sort_values(by='Importance score', ascending=False)

sns.barplot(x='Importance score', y='Features', data=imp_df, ax=ax, palette='rocket')
ax.set_title('CatBoost Feature Importance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("Mencetak Prediksi untuk 21.780 baris Test Set...")
X_test = test_full[features]
test_pool = Pool(X_test, cat_features=cat_features)
test_full['tma_mdpl'] = model_final.predict(test_pool)

sub_pred = test_full[['id', 'tma_mdpl']].copy()
sub = pd.merge(sample_sub[['id']], sub_pred, on='id', how='left')
sub['tma_mdpl'] = sub['tma_mdpl'].fillna(method='ffill').fillna(method='bfill')

output_file = 'submission_V51_CatBoost_Universe.csv'
sub.to_csv(output_file, index=False)
print(f"✅ Selesai! File '{output_file}' siap disubmit! CatBoost siap mendominasi!")""")
]

with open("c:\\Users\\LENOVO X1 CARBON\\Documents\\SSF-SSDS\\V51_CatBoost_Universe.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
