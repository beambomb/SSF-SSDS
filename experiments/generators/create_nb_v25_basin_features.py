import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

def create_v25_notebook():
    nb = new_notebook()
    
    cells = [
        new_markdown_cell('# 🌊 V25 BASIN-WIDE SPATIAL FEATURES (BANJIR KIRIMAN) 🌊\\nDi V25 ini, kita kembali menggunakan **Tweedie Loss** (karena eksperimen Huber membuktikan Tweedie lebih superior untuk skor RMSE). Namun, untuk mengatasi stasiun liar yang underfit seperti Jarum dan Bojonegoro, kita menyuntikkan **Fitur Global (Basin-Wide)**!\\n\\nSetiap model lokal kini tidak hanya melihat cuaca di stasiunnya sendiri, tapi juga melihat total curah hujan dari **seluruh DAS Bengawan Solo**. Ini memungkinkan Bojonegoro (Hilir) bersiap menerima banjir kiriman ketika curah hujan agregat di Hulu sedang ekstrem!'),
        
        new_markdown_cell('## 1. Persiapan Environment & Pemuatan Data'),
        new_code_cell('''!pip install -q xgboost scikit-learn optuna folium
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

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

        new_markdown_cell('## 2. Feature Engineering (LOCAL + GLOBAL BASIN)\\nDi sinilah keajaiban terjadi. Kita menambahkan fitur `basin_rain_sum` yang merepresentasikan total curah hujan seluruh jaringan sungai Bengawan Solo!'),
        new_code_cell('''print("Mengekstraksi Fitur Cuaca Lokal & GLOBAL BASIN...")
lingkungan_df = lingkungan_df.sort_values(['nama_pos', 'datetime']).reset_index(drop=True)

def build_exogenous_features(env_df):
    agg_df = env_df[['datetime', 'nama_pos']].copy()
    
    # 📌 1. FITUR LOKAL (Curah hujan di titik pos itu sendiri)
    grouped = env_df.groupby('nama_pos')
    
    if 'rainfall_mm' in env_df.columns:
        agg_df['rain_sum_6h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(6, min_periods=1).sum())
        agg_df['rain_sum_72h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(72, min_periods=1).sum())
        agg_df['rain_sum_168h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(168, min_periods=1).sum())
        agg_df['rain_sum_360h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(360, min_periods=1).sum())
        agg_df['rain_sum_720h'] = grouped['rainfall_mm'].transform(lambda x: x.rolling(720, min_periods=1).sum())
        
    if 'soil_moisture_0_7cm' in env_df.columns:
        agg_df['soil_mean_6h'] = grouped['soil_moisture_0_7cm'].transform(lambda x: x.rolling(6, min_periods=1).mean())
        agg_df['soil_mean_72h'] = grouped['soil_moisture_0_7cm'].transform(lambda x: x.rolling(72, min_periods=1).mean())
        agg_df['soil_mean_168h'] = grouped['soil_moisture_0_7cm'].transform(lambda x: x.rolling(168, min_periods=1).mean())
        
    for c in ['nino_34', 'mjo_phase']:
        if c in env_df.columns:
            agg_df[f'{c}_mean_720h'] = grouped[c].transform(lambda x: x.rolling(720, min_periods=1).mean())
            
    for c in ['temperature_c', 'wind_speed_kmh', 'surface_pressure_hpa', 'humidity_pct']:
        if c in env_df.columns:
            agg_df[f'{c}_mean_6h'] = grouped[c].transform(lambda x: x.rolling(6, min_periods=1).mean())

    # 📌 2. FITUR GLOBAL BASIN (Indikator Banjir Kiriman dari Hulu)
    print("Menghitung Agregat DAS Bengawan Solo...")
    basin_env = env_df.groupby('datetime').agg({
        'rainfall_mm': 'mean', # Rata-rata hujan di seluruh DAS pada jam tersebut
        'soil_moisture_0_7cm': 'mean'
    }).reset_index()

    # Efek akumulasi memori untuk seluruh sungai
    basin_env['basin_rain_sum_72h'] = basin_env['rainfall_mm'].rolling(72, min_periods=1).sum()
    basin_env['basin_rain_sum_168h'] = basin_env['rainfall_mm'].rolling(168, min_periods=1).sum()
    basin_env['basin_rain_sum_360h'] = basin_env['rainfall_mm'].rolling(360, min_periods=1).sum()
    basin_env['basin_soil_mean_168h'] = basin_env['soil_moisture_0_7cm'].rolling(168, min_periods=1).mean()
    
    basin_features = basin_env[['datetime', 'basin_rain_sum_72h', 'basin_rain_sum_168h', 'basin_rain_sum_360h', 'basin_soil_mean_168h']]
    
    # 📌 3. MERGE LOKAL DAN GLOBAL
    agg_df = pd.merge(agg_df, basin_features, on='datetime', how='left')

    return agg_df

env_feat = build_exogenous_features(lingkungan_df)

def prepare_dataset(base_df, env_feat):
    df = base_df.copy()
    
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['dayofyear'] = df['datetime'].dt.dayofyear
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12.0)
    
    df['hour_sin_12h'] = np.sin(2 * np.pi * df['hour']/12.0)
    df['hour_cos_12h'] = np.cos(2 * np.pi * df['hour']/12.0)
    
    df = pd.merge(df, env_feat, on=['datetime', 'nama_pos'], how='left')
    return df

print("Menyusun Train & Test Set...")
train_full = prepare_dataset(train_df, env_feat)

test_parsed = test_df.copy()
split_info = test_parsed['id'].str.split(' - ', n=1, expand=True)
test_parsed['datetime'] = pd.to_datetime(split_info[0])
test_parsed['nama_pos'] = split_info[1]
test_full = prepare_dataset(test_parsed, env_feat)

train_full = train_full.dropna(subset=['tma_mdpl']).reset_index(drop=True)

drop_cols = ['datetime', 'nama_pos', 'tma_mdpl', 'id']
features = [c for c in train_full.columns if c not in drop_cols]
print(f"Total Fitur: {len(features)}")
print(f"Daftar Fitur: {features}")
'''),

        new_markdown_cell('## 3. Optuna: Tweedie Loss dengan Fitur Basin\\nModel sekarang punya pandangan helikopter (Helicopter View) terhadap cuaca seluruh DAS!'),
        new_code_cell('''stations = train_full['nama_pos'].unique()
station_models = {}
station_scores = {}

optuna.logging.set_verbosity(optuna.logging.ERROR)
N_TRIALS = 15

print(f"Memulai Optuna (Tweedie) untuk 30 Stasiun (Total {30 * N_TRIALS} Trials)...\\n")

for st in stations:
    X_st = train_full[train_full['nama_pos'] == st][features]
    y_st = train_full[train_full['nama_pos'] == st]['tma_mdpl']
    
    def objective(trial):
        params = {
            'objective': 'reg:tweedie', 
            'tweedie_variance_power': trial.suggest_float('tweedie_variance_power', 1.1, 1.9),
            'eval_metric': 'rmse', 
            'tree_method': 'hist',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'n_estimators': 800,
            'random_state': 42
        }
        
        split_idx = int(len(X_st) * 0.8)
        X_tr, y_tr = X_st.iloc[:split_idx], np.clip(y_st.iloc[:split_idx], 0.0001, None)
        X_va, y_va = X_st.iloc[split_idx:], np.clip(y_st.iloc[split_idx:], 0.0001, None)
        
        model = xgb.XGBRegressor(**params, early_stopping_rounds=30)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        
        preds = model.predict(X_va)
        return root_mean_squared_error(y_va, preds)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS)
    
    best_params = study.best_params
    best_params['objective'] = 'reg:tweedie'
    best_params['eval_metric'] = 'rmse'
    best_params['tree_method'] = 'hist'
    best_params['n_estimators'] = 800
    best_params['random_state'] = 42

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    models = []
    scores = []
    
    for train_idx, val_idx in kf.split(X_st, y_st):
        X_tr, y_tr = X_st.iloc[train_idx], np.clip(y_st.iloc[train_idx], 0.0001, None)
        X_va, y_va = X_st.iloc[val_idx], np.clip(y_st.iloc[val_idx], 0.0001, None)
        
        model = xgb.XGBRegressor(**best_params, early_stopping_rounds=50)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        
        val_preds = model.predict(X_va)
        scores.append(root_mean_squared_error(y_va, val_preds))
        models.append(model)
        
    avg_rmse = np.mean(scores)
    station_models[st] = models
    station_scores[st] = avg_rmse
    
    print(f"[{st}] - Tweedie Power: {best_params['tweedie_variance_power']:.2f} | Eval RMSE: {avg_rmse:.4f}")

global_rmse = np.mean(list(station_scores.values()))
print(f"\\n🏆 GRAND AVERAGE RMSE ACROSS 30 STATIONS: {global_rmse:.4f}")
'''),

        new_markdown_cell('## 4. Visualisasi & Prediksi Akhir\\nPerhatikan Feature Importance! Apakah fitur `basin_rain_sum_168h` masuk ke jajaran top 10 untuk stasiun Hilir?'),
        new_code_cell('''TARGET_STATION = "Bojonegoro - Kali Kethek" 

fig, ax = plt.subplots(1, 2, figsize=(22, 8))

feat_imp = np.mean([m.feature_importances_ for m in station_models[TARGET_STATION]], axis=0)
imp_df = pd.DataFrame({'Features': features, 'Importance score': feat_imp})
imp_df = imp_df.sort_values(by='Importance score', ascending=False).head(20)

sns.barplot(x='Importance score', y='Features', data=imp_df, ax=ax[0], palette='viridis')
ax[0].set_title(f'Top 20 Feature Importance [{TARGET_STATION}]')

X_target = train_full[train_full['nama_pos'] == TARGET_STATION][features]
y_target = train_full[train_full['nama_pos'] == TARGET_STATION]['tma_mdpl']

val_preds = np.zeros(len(X_target))
for m in station_models[TARGET_STATION]:
    val_preds += m.predict(X_target) / 5

sub_actual = y_target.iloc[-200:].values
sub_pred = val_preds[-200:]
ax[1].plot(sub_actual, label='Actual TMA', color='blue', alpha=0.7, linewidth=2)
ax[1].plot(sub_pred, label='Predicted (Tweedie + Basin)', color='red', alpha=0.7, linewidth=2)
ax[1].set_title(f'Actual vs Predicted - 200 Hari Terakhir [{TARGET_STATION}]')
ax[1].legend()

plt.tight_layout()
plt.show()
'''),
        new_code_cell('''test_preds = np.zeros(len(test_full))

for st in stations:
    idx = test_full['nama_pos'] == st
    if idx.sum() == 0: continue
        
    X_test_st = test_full.loc[idx, features]
    
    st_preds = np.zeros(len(X_test_st))
    for m in station_models[st]:
        st_preds += m.predict(X_test_st) / 5
        
    test_full.loc[idx, 'tma_mdpl'] = st_preds

sub_pred = test_full[['id', 'tma_mdpl']].copy()
sub = pd.merge(sample_sub[['id']], sub_pred, on='id', how='left')

sub['tma_mdpl'] = sub['tma_mdpl'].fillna(method='ffill').fillna(method='bfill')

display(sub.head())
sub.to_csv('submission_v25_basin_tweedie.csv', index=False)
print("\\n✅ File 'submission_v25_basin_tweedie.csv' berhasil dibuat dengan kekuatan Fitur DAS Global!")
''')
    ]
    
    nb.cells.extend(cells)
    with open('V25_Basin_Features_Tweedie.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == '__main__':
    create_v25_notebook()
    print("Notebook V25_Basin_Features_Tweedie.ipynb berhasil dibuat!")
