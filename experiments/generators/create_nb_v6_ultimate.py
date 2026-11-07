import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()

nb.cells.extend([
    new_markdown_cell('# 🌪️ V6: The Ultimate Exogenous Features & Holy Trinity\n\nNotebook ini adalah puncak evolusi dari strategi *Direct Exogenous*. Kita menginjeksi **Fitur Hidrologi Tingkat Dewa** (Runoff Proxy, Soil Gradient, MJO Kinematics, Wind Cyclical, dan Landcover) sambil tetap mempertahankan arsitektur Ensemble 3 Model (XGB + LGBM + Cat) tanpa Target Lag untuk jaminan 0% *Error Compounding*.'),
    
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

    new_markdown_cell('## 3. Ekstraksi Fitur (The Ultimate Hydrological Features) 💧🌪️\n\nKita menerapkan saran riset *State-of-the-Art*:\n1. **Runoff Proxy:** `rainfall * soil_moisture`\n2. **Soil Gradient:** Daya serap tanah per jam\n3. **MJO Speed:** Kinematika iklim global `sqrt(rmm1^2 + rmm2^2)`\n4. **Cyclical Wind:** `sin` dan `cos` dari arah angin\n5. **Landcover & Built Surface:** Dimasukkan sebagai fitur spasial.'),
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
        agg_df['mjo_speed_mean_720h'] = grouped['mjo_speed'].transform(lambda x: x.rolling(720, min_periods=1).mean())
        
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

# Filter HANYA ke waktu pencatatan (06:00, 12:00, 18:00) untuk target
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

# Label Encoding Nama Pos & Landcover Name
le_pos = LabelEncoder()
all_stations = pd.concat([train_full['nama_pos'], test_full['nama_pos']]).unique()
train_full['nama_pos_encoded'] = le_pos.fit_transform(train_full['nama_pos'])
test_full['nama_pos_encoded'] = le_pos.transform(test_full['nama_pos'])

if 'landcover_name' in train_full.columns:
    le_land = LabelEncoder()
    # Fillna just in case
    train_full['landcover_name'] = train_full['landcover_name'].fillna('Unknown')
    test_full['landcover_name'] = test_full['landcover_name'].fillna('Unknown')
    all_land = pd.concat([train_full['landcover_name'], test_full['landcover_name']]).unique()
    train_full['landcover_encoded'] = le_land.fit_transform(train_full['landcover_name'])
    test_full['landcover_encoded'] = le_land.transform(test_full['landcover_name'])

train_full = train_full.dropna(subset=['tma_mdpl']).reset_index(drop=True)

# Definisikan kolom fitur (Kecualikan string dan identifier)
drop_cols = ['datetime', 'nama_pos', 'tma_mdpl', 'id', 'landcover_name']
features = [c for c in train_full.columns if c not in drop_cols]
print(f"Total Fitur Ultimate: {len(features)}")
'''),

    new_markdown_cell('## 4. Log Transformation & Setup X, y'),
    new_code_cell('''X = train_full[features]
y_asli = train_full['tma_mdpl']

# Log transform target
y_log = np.log1p(y_asli)

print("Target has been Log-Transformed!")
'''),

    new_markdown_cell('## 5. Optuna Tuning (XGBoost)'),
    new_code_cell('''RUN_OPTUNA = False # Ubah ke True jika Anda ingin mencari parameter terbaik (~30 menit)

if RUN_OPTUNA:
    def objective(trial):
        param = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'hist',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 5, 9),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 5.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 5.0, log=True),
            'random_state': 42
        }
        split_idx = int(len(X) * 0.8)
        dtrain = xgb.DMatrix(X.iloc[:split_idx], label=y_log.iloc[:split_idx])
        dval = xgb.DMatrix(X.iloc[split_idx:], label=y_log.iloc[split_idx:])
        model = xgb.train(param, dtrain, num_boost_round=1000, evals=[(dval, 'val')], early_stopping_rounds=30, verbose_eval=False)
        return root_mean_squared_error(y_log.iloc[split_idx:], model.predict(dval))

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50) 
    xgb_params = study.best_params
else:
    xgb_params = {'learning_rate': 0.03, 'max_depth': 8, 'subsample': 0.8}

xgb_params.update({'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'tree_method': 'hist', 'random_state': 42})
'''),

    new_markdown_cell('## 6. Training "The Kaggle Holy Trinity" (K-Fold)'),
    new_code_cell('''kf = KFold(n_splits=5, shuffle=True, random_state=42)

xgb_models = []
lgb_models = []
cat_models = []

lgb_params = {
    'objective': 'regression', 'metric': 'rmse', 'learning_rate': 0.03,
    'num_leaves': 64, 'max_depth': 8, 'feature_fraction': 0.8,
    'bagging_fraction': 0.8, 'bagging_freq': 1, 'verbose': -1, 'random_state': 42
}

# Tentukan mana fitur yang berbentuk kategori
cat_features_cols = ['nama_pos_encoded']
if 'landcover_encoded' in features:
    cat_features_cols.append('landcover_encoded')
    
lgb_cat_idx = [features.index(c) for c in cat_features_cols]
catboost_cat_idx = lgb_cat_idx.copy()

print("Mulai melatih TRINITY K-FOLD (XGB, LGBM, CAT)...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_log)):
    print(f"\\n========== FOLD {fold+1} ==========")
    X_tr, y_tr = X.iloc[train_idx], y_log.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y_log.iloc[val_idx]
    
    # 1. XGBoost
    dtrain_xgb = xgb.DMatrix(X_tr, label=y_tr)
    dval_xgb = xgb.DMatrix(X_va, label=y_va)
    model_xgb = xgb.train(xgb_params, dtrain_xgb, num_boost_round=3000, 
                          evals=[(dtrain_xgb, 'train'), (dval_xgb, 'val')],
                          early_stopping_rounds=50, verbose_eval=500)
    xgb_models.append(model_xgb)
    
    # 2. LightGBM
    dtrain_lgb = lgb.Dataset(X_tr, label=y_tr, categorical_feature=lgb_cat_idx)
    dval_lgb = lgb.Dataset(X_va, label=y_va, reference=dtrain_lgb, categorical_feature=lgb_cat_idx)
    model_lgb = lgb.train(lgb_params, dtrain_lgb, num_boost_round=3000,
                          valid_sets=[dtrain_lgb, dval_lgb],
                          callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=500)])
    lgb_models.append(model_lgb)
    
    # 3. CatBoost
    model_cat = CatBoostRegressor(iterations=3000, learning_rate=0.03, depth=8,
                                  eval_metric='RMSE', random_seed=42, verbose=500)
    model_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), 
                  cat_features=catboost_cat_idx, early_stopping_rounds=50)
    cat_models.append(model_cat)
'''),

    new_markdown_cell('## 7. Ensemble Inference (Blending) ⚔️'),
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

# KEMBALIKAN NILAI DARI LOG KE SKALA ASLI
preds_xgb_asli = np.expm1(preds_xgb)
preds_lgb_asli = np.expm1(preds_lgb)
preds_cat_asli = np.expm1(preds_cat)

# BLENDING THE TRINITY: 40% XGBoost, 30% LightGBM, 30% CatBoost
final_preds = (0.4 * preds_xgb_asli) + (0.3 * preds_lgb_asli) + (0.3 * preds_cat_asli)

test_full['tma_mdpl'] = final_preds

# Format Submission
sub_pred = test_full[['id', 'tma_mdpl']].copy()
sub = pd.merge(sample_sub[['id']], sub_pred, on='id', how='left')
sub['tma_mdpl'] = sub['tma_mdpl'].fillna(method='ffill').fillna(method='bfill')

display(sub.head(10))
sub.to_csv('submission_v6_ultimate.csv', index=False)
print("\\n✅ THE ULTIMATE V6 COMPLETED! File 'submission_v6_ultimate.csv' siap di-submit!")
''')
])

with open('V6_Ultimate_TMA.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
