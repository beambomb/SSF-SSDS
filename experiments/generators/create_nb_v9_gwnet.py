import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()

nb.cells.extend([
    new_markdown_cell('# 🌌 V9: Graph WaveNet (GWNet) untuk Peramalan Tinggi Muka Air Spatio-Temporal\n\nNotebook ini mengimplementasikan arsitektur **Graph WaveNet** sebagai pengembangan dari ST-GCN. Pengembangan utama mencakup:\n1. **Koreksi Arah Topologi:** Memastikan agregasi spasial dilakukan dari hulu ke hilir.\n2. **Bobot Jarak Topologis:** Menerapkan bobot ketetanggaan berdasarkan jarak *network hops* (invers jarak).\n3. **Dilated Convolutions & Adaptive Matrix:** Penggunaan konvolusi temporal merenggang untuk menangkap *delay* aliran air, serta matriks ketetanggaan adaptif untuk mempelajari relasi spasial yang tidak tertangkap oleh topologi eksplisit.'),
    
    new_code_cell('''!pip install -q pandas numpy matplotlib seaborn scikit-learn tensorflow geopandas networkx shapely'''),
    
    new_markdown_cell('## 1. Impor Pustaka dan Pengaturan Awal\nBagian ini memuat pustaka standar untuk manipulasi data, visualisasi spasial, serta pembangunan arsitektur *deep learning* menggunakan TensorFlow.'),
    new_code_cell('''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc
import warnings

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, Layer, Permute, Reshape, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error

import geopandas as gpd
from shapely.geometry import Point
import networkx as nx

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 150)
sns.set_theme(style="whitegrid")

# Reproduksibilitas
np.random.seed(42)
tf.random.set_seed(42)
'''),

    new_markdown_cell('## 2. Pemuatan Data\nMemuat data utama dan data pendukung. Dataset mencakup fitur cuaca makro/mikro, elevasi (MDPL), dan titik koordinat stasiun.'),
    new_code_cell('''import os

# =========================================================================
# OPSI 1: MENGGUNAKAN GDOWN (Direkomendasikan untuk Kaggle/Colab tanpa Mount)
# Jika kamu menyimpan dataset dalam bentuk ZIP di Google Drive, 
# masukkan ID file ZIP tersebut di bawah ini dan uncomment barisnya.
# =========================================================================
# !pip install -q gdown
# !gdown --id "MASUKKAN_ID_FILE_ZIP_GOOGLE_DRIVE_DISINI"
# !unzip -q -o dataset.zip -d ./dataset_folder/
# path = './dataset_folder/'

# =========================================================================
# OPSI 2: AUTO-DETECT PATH (Kaggle Dataset Native / Colab Mount / Lokal)
# =========================================================================
if not 'path' in locals(): # Jika opsi 1 tidak digunakan
    if os.path.exists('/kaggle/input'):
        print("Mendeteksi environment Kaggle...")
        dataset_folders = [f for f in os.listdir('/kaggle/input') if os.path.isdir(f'/kaggle/input/{f}')]
        if dataset_folders:
            # Mencari folder yang berisi train.csv
            path = f'/kaggle/input/{dataset_folders[0]}/'
        else:
            path = '/kaggle/input/'
        output_path = '/kaggle/working/'
    elif os.path.exists('/content'):
        print("Mendeteksi environment Google Colab...")
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            path = '/content/drive/MyDrive/ssf-ssds/'
        except:
            path = '/content/'
        output_path = '/content/'
    else:
        print("Mendeteksi environment Lokal...")
        path = './'
        output_path = './'

print(f"Path dataset aktif: {path}")

# Membaca Data
train_df = pd.read_csv(path + 'train.csv', parse_dates=['datetime'])
test_df = pd.read_csv(path + 'test.csv')
sample_sub = pd.read_csv(path + 'sample_submission.csv')

lingkungan_df = pd.read_csv(path + 'data_pendukung/data_lingkungan.csv', parse_dates=['datetime'])
koordinat_df = pd.read_csv(path + 'data_pendukung/koordinat_pos.csv')
'''),

    new_markdown_cell('## 3. Analisis Data Eksploratif (EDA)\nEvaluasi pola dasar dalam data untuk memvalidasi penggunaan pendekatan Spatio-Temporal.'),
    
    new_code_cell('''# Persiapan dataset gabungan untuk EDA
train_eda = pd.merge(train_df, koordinat_df, on='nama_pos', how='left')
env_eda = lingkungan_df[lingkungan_df['datetime'].dt.hour.isin([6,12,18])].copy()
merged_eda = pd.merge(train_eda, env_eda, on=['datetime', 'nama_pos'], how='inner')

# EDA 1: Distribusi Target (Tinggi Muka Air)
plt.figure(figsize=(10, 5))
sns.histplot(train_eda['tma_mdpl'], bins=100, kde=True, color='teal')
plt.title('Distribusi Elevasi Tinggi Muka Air (TMA)')
plt.xlabel('Tinggi Muka Air (MDPL)')
plt.ylabel('Frekuensi')
plt.show()
'''),
    new_markdown_cell('**Analisis:** Distribusi target bersifat *right-skewed*. Mayoritas pengukuran terkonsentrasi pada elevasi rendah (hilir sungai), sementara sebagian kecil pengukuran memiliki elevasi di atas 100 MDPL (stasiun pegunungan/hulu). Hal ini menegaskan perlunya standardisasi target sebelum pelatihan model.'),

    new_code_cell('''# EDA 2: Variabilitas Spasial Antar Stasiun
plt.figure(figsize=(15, 6))
mean_elev = train_eda.groupby('nama_pos')['tma_mdpl'].mean().sort_values(ascending=False)
sns.barplot(x=mean_elev.index, y=mean_elev.values, palette='viridis')
plt.xticks(rotation=45, ha='right')
plt.title('Rata-rata Elevasi TMA per Stasiun (Hulu ke Hilir)')
plt.ylabel('Rata-rata TMA (MDPL)')
plt.show()
'''),
    new_markdown_cell('**Analisis:** Terdapat disparitas elevasi yang sangat signifikan antar stasiun. Air secara fisik akan mengalir dari stasiun di sisi kiri grafik (Hulu) menuju ke sisi kanan grafik (Hilir). Pemodelan ketergantungan arah (*directed graph*) sangat esensial.'),

    new_code_cell('''# EDA 3: Deteksi Outlier (Kejadian Banjir)
plt.figure(figsize=(15, 6))
random_sts = np.random.choice(train_eda['nama_pos'].unique(), 10, replace=False)
sns.boxplot(data=train_eda[train_eda['nama_pos'].isin(random_sts)], x='nama_pos', y='tma_mdpl')
plt.title('Deteksi Nilai Ekstrem (Banjir) per Stasiun Sampel')
plt.xticks(rotation=45)
plt.show()

# Pembersihan memori
del train_eda, env_eda, merged_eda
gc.collect()
'''),
    new_markdown_cell('**Analisis:** Setiap stasiun memiliki nilai observasi yang jauh melampaui rentang *interquartile*. Outlier ini merepresentasikan fenomena banjir ekstrem yang merupakan tantangan utama dalam prediksi regresi.'),

    new_markdown_cell('## 4. Ekstraksi Topologi Spasial Berarah (*Directed Weighted Adjacency Matrix*)\nProsedur ini mengekstrak hubungan hulu-hilir antar stasiun menggunakan data `HydroRIVERS`. Perbaikan dari iterasi sebelumnya meliputi penentuan arah agregasi informasi (hulu ke hilir) dan pemberian bobot atenuasi berdasarkan jarak hop jaringan.'),
    new_code_cell('''print("Membangun Matriks Ketetanggaan Berarah (Directed Weighted Graph)...")

station_names = sorted(koordinat_df['nama_pos'].unique())
NUM_STATIONS = len(station_names)
A = np.zeros((NUM_STATIONS, NUM_STATIONS), dtype=np.float32)

try:
    rivers_shp_path = path + 'data_pendukung/HydroRIVERS_v10_au_shp/HydroRIVERS_v10_au.shp'
    rivers = gpd.read_file(rivers_shp_path)
    
    gdf_coords = gpd.GeoDataFrame(
        koordinat_df, geometry=gpd.points_from_xy(koordinat_df.longitude, koordinat_df.latitude), crs="EPSG:4326"
    )
    if rivers.crs != "EPSG:4326":
        gdf_coords = gdf_coords.to_crs(rivers.crs)
        
    matched = gpd.sjoin_nearest(gdf_coords, rivers, how='left')
    station_to_river = {row['nama_pos']: row['HYRIV_ID'] for _, row in matched.iterrows()}
        
    # Membangun Graf Jaringan Sungai
    G = nx.DiGraph()
    for _, row in rivers.iterrows():
        G.add_edge(row['HYRIV_ID'], row['NEXT_DOWN'])
        
    # Membangun Weighted Adjacency Matrix
    for i, st_source in enumerate(station_names):
        river_source = station_to_river.get(st_source)
        for j, st_target in enumerate(station_names):
            if i != j:
                river_target = station_to_river.get(st_target)
                if river_source and river_target:
                    try:
                        # Menghitung jarak hop topologis
                        hops = nx.shortest_path_length(G, source=river_source, target=river_target)
                        
                        # Rumus atenuasi jarak (inverse distance weight)
                        weight = 1.0 / (hops + 1.0)
                        
                        # PENYELESAIAN MASALAH ARAH: 
                        # Node target (Hilir) harus menyerap informasi dari node source (Hulu).
                        # Dalam operasi GCN (Output_w = A_vw * Input_v), A_vw melambangkan arah v -> w.
                        # Indeks pertama A mewakili source (v), indeks kedua mewakili target (w).
                        A[i, j] = weight 
                    except nx.NetworkXNoPath:
                        pass
    
    print("Topologi Berarah Berhasil Diekstrak!")

except Exception as e:
    print(f"Ekstraksi gagal ({e}). Menggunakan Fallback Matrix (Geographic Distance)...")
    from sklearn.metrics.pairwise import haversine_distances
    coords_rad = np.radians(koordinat_df.set_index('nama_pos').loc[station_names][['latitude', 'longitude']])
    dist = haversine_distances(coords_rad) * 6371
    # Matriks fallback spasial standar
    A = np.exp(-(dist**2) / 100.0) 
    np.fill_diagonal(A, 0.0)

# Normalisasi Matriks berbasis Kolom (Karena agregasi menuju target kolom)
col_sums = A.sum(axis=0)
col_sums[col_sums == 0] = 1.0
A_norm = A / col_sums[np.newaxis, :]

print(f"Dimensi Matriks Ketetanggaan: {A_norm.shape}")
'''),

    new_markdown_cell('## 5. Prapemrosesan Tensor Spatio-Temporal\nMelakukan *resampling* interval reguler dan penyusunan struktur tensor 4-dimensi: `(Batch, Waktu, Spasial, Fitur)`.'),
    new_code_cell('''print("Memproses Fitur Cuaca menjadi Tensor Spatio-Temporal...")

# Ekstraksi Fitur Waktu (Siklikal) untuk menangkap Pasang Surut & Musim
lingkungan_df['hour'] = lingkungan_df['datetime'].dt.hour
lingkungan_df['month'] = lingkungan_df['datetime'].dt.month

lingkungan_df['hour_sin'] = np.sin(2 * np.pi * lingkungan_df['hour'] / 24.0)
lingkungan_df['hour_cos'] = np.cos(2 * np.pi * lingkungan_df['hour'] / 24.0)
lingkungan_df['month_sin'] = np.sin(2 * np.pi * lingkungan_df['month'] / 12.0)
lingkungan_df['month_cos'] = np.cos(2 * np.pi * lingkungan_df['month'] / 12.0)

env_features = ['rainfall_mm', 'soil_moisture_0_7cm', 'temperature_c', 'humidity_pct', 'surface_pressure_hpa', 'wind_speed_kmh', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos']
if 'rmm1' in lingkungan_df.columns:
    lingkungan_df['mjo_speed'] = np.sqrt(lingkungan_df['rmm1']**2 + lingkungan_df['rmm2']**2)
    env_features += ['mjo_speed', 'nino_34']

# Proxy Runoff & Akumulasi Hujan
lingkungan_df['runoff_proxy'] = lingkungan_df['rainfall_mm'] * lingkungan_df['soil_moisture_0_7cm']
env_features += ['runoff_proxy', 'rainfall_roll_24h', 'rainfall_roll_72h', 'soil_moisture_diff']

# Agregasi ke Resolusi 6-Jam
env_6h_list = []
for station, group in lingkungan_df.groupby('nama_pos'):
    group = group.set_index('datetime').sort_index()
    resampled = group[['soil_moisture_0_7cm', 'temperature_c', 'humidity_pct', 'surface_pressure_hpa', 'wind_speed_kmh', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos'] + ([f for f in ['mjo_speed', 'nino_34'] if f in group.columns])].resample('6H').mean()
    
    resampled['rainfall_mm'] = group['rainfall_mm'].resample('6H').sum()
    resampled['runoff_proxy'] = group['runoff_proxy'].resample('6H').sum()
    
    # Feature Engineering Hidrologi Lanjutan
    resampled['rainfall_roll_24h'] = resampled['rainfall_mm'].rolling(window=4, min_periods=1).sum()
    resampled['rainfall_roll_72h'] = resampled['rainfall_mm'].rolling(window=12, min_periods=1).sum()
    resampled['soil_moisture_diff'] = resampled['soil_moisture_0_7cm'].diff().fillna(0)
    
    resampled['nama_pos'] = station
    env_6h_list.append(resampled.reset_index())

env_6h = pd.concat(env_6h_list, ignore_index=True).dropna().reset_index(drop=True)

# Standardisasi
scaler = StandardScaler()
env_6h[env_features] = scaler.fit_transform(env_6h[env_features])

# Pivot Format
env_pivot = env_6h.pivot(index='datetime', columns='nama_pos', values=env_features)
for feat in env_features:
    env_pivot[feat] = env_pivot[feat].reindex(columns=station_names, fill_value=0)

train_df['tma_mdpl'] = train_df['tma_mdpl'].astype(np.float32)
target_pivot = train_df.pivot(index='datetime', columns='nama_pos', values='tma_mdpl')
target_pivot = target_pivot.reindex(columns=station_names)

SEQ_LEN = 30
NUM_FEATURES = len(env_features)

def build_sequences(target_times, env_pivot_df, target_pivot_df=None):
    X, y, valid = [], [], []
    env_times = env_pivot_df.index
    env_tensor = np.zeros((len(env_times), NUM_STATIONS, NUM_FEATURES), dtype=np.float32)
    
    for f_idx, feat in enumerate(env_features):
        env_tensor[:, :, f_idx] = env_pivot_df[feat].values
        
    for t in target_times:
        try:
            t_idx = env_times.get_loc(t)
            if t_idx >= SEQ_LEN - 1:
                X.append(env_tensor[t_idx - SEQ_LEN + 1 : t_idx + 1])
                if target_pivot_df is not None:
                    y.append(target_pivot_df.loc[t].values)
                valid.append(t)
        except KeyError:
            pass
            
    return np.array(X), (np.array(y) if target_pivot_df is not None else None), valid

train_times = target_pivot.dropna(how='all').index.sort_values().unique()
X_train_full, y_train_full, _ = build_sequences(train_times, env_pivot, target_pivot)

test_parsed = test_df.copy()
test_parsed['datetime'] = pd.to_datetime(test_parsed['id'].str.split(' - ', n=1, expand=True)[0])
X_test_full, _, test_valid_times = build_sequences(test_parsed['datetime'].sort_values().unique(), env_pivot)

# Pembersihan baris target yang sepenuhnya NaN
valid_mask = ~np.isnan(y_train_full).all(axis=1)
X_train_full = X_train_full[valid_mask]
y_train_full = y_train_full[valid_mask]

# Standardisasi Target 
target_mean = np.nanmean(y_train_full, axis=0)
target_std = np.nanstd(y_train_full, axis=0) + 1e-8
y_train_scaled = (y_train_full - target_mean) / target_std

# Penggantian nilai target yang masih NaN dengan 0 (nilai rata-rata setelah skala)
y_train_scaled = np.nan_to_num(y_train_scaled, nan=0.0)

print(f"Struktur Tensor Input: {X_train_full.shape}")
print(f"Struktur Tensor Target: {y_train_scaled.shape}")
'''),

    new_markdown_cell('## 6. Arsitektur Graph WaveNet (Kustom)\nArsitektur ini mendefinisikan blok konvolusi spasial dan temporal secara bergantian. Pendekatan adaptif diterapkan pada layer GCN untuk memperbaiki defisiensi pada abstraksi topologi eksplisit.'),
    new_code_cell('''import keras
from keras import ops

class TemporalConvNet(Layer):
    """Layer konvolusi temporal merenggang (Dilated CNN) untuk menangkap dependensi jangka panjang."""
    def __init__(self, filters, kernel_size=2, dilation_rate=1, **kwargs):
        super(TemporalConvNet, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.conv = Conv1D(filters=filters, kernel_size=kernel_size, 
                           dilation_rate=dilation_rate, padding='causal', activation='relu')

    def call(self, inputs):
        # Struktur masuk: (Batch, Nodes, Time, Features)
        b = ops.shape(inputs)[0]
        n = NUM_STATIONS
        t = SEQ_LEN
        f = inputs.shape[-1]
        
        # Penyatuan batch dan nodes untuk konvolusi 1D paralel
        x = ops.reshape(inputs, [-1, t, f])
        x = self.conv(x)
        
        # Pengembalian ke struktur awal
        return ops.reshape(x, [-1, n, t, self.filters])
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.filters)

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "dilation_rate": self.dilation_rate,
        })
        return config

class AdaptiveGraphConv(Layer):
    """Layer konvolusi graf yang menggabungkan matriks spasial apriori dan matriks adaptif yang dipelajari."""
    def __init__(self, A_matrix, embed_dim=10, **kwargs):
        super(AdaptiveGraphConv, self).__init__(**kwargs)
        self.A_matrix = A_matrix.tolist() if isinstance(A_matrix, np.ndarray) else A_matrix
        self.embed_dim = embed_dim

    def build(self, input_shape):
        num_nodes = NUM_STATIONS
        self.A = ops.convert_to_tensor(self.A_matrix, dtype='float32')
        self.E1 = self.add_weight(shape=(num_nodes, self.embed_dim), initializer='random_normal', trainable=True)
        self.E2 = self.add_weight(shape=(num_nodes, self.embed_dim), initializer='random_normal', trainable=True)
        super(AdaptiveGraphConv, self).build(input_shape)

    def call(self, inputs):
        # Pembentukan Adaptive Matrix via perkalian embedding
        A_adapt = ops.matmul(self.E1, ops.transpose(self.E2))
        A_adapt = keras.activations.relu(A_adapt)
        A_adapt = keras.activations.softmax(A_adapt, axis=1) 
        
        A_combined = self.A + A_adapt
        
        # Operasi Graph Convolution (Aggregasi spasial v -> w)
        # inputs shape: (Batch, Nodes, Time, Features)
        return ops.einsum('vw,bvtf->bwtf', A_combined, inputs)
        
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            "A_matrix": self.A_matrix,
            "embed_dim": self.embed_dim,
        })
        return config

@keras.saving.register_keras_serializable()
def asymmetric_mse(y_true, y_pred):
    # Loss custom untuk menyembuhkan sifat "malu-malu" model.
    # Memberi penalti 3x lebih besar jika model menebak terlalu rendah (under-prediction saat banjir).
    error = y_true - y_pred
    return ops.mean(ops.where(error > 0, 3.0 * ops.square(error), ops.square(error)))

def build_gwnet(A_matrix):
    inputs = Input(shape=(SEQ_LEN, NUM_STATIONS, NUM_FEATURES))
    
    # Restrukturisasi ke (Batch, Nodes, Time, Features)
    x = Permute((2, 1, 3))(inputs)
    
    # Layer Spatio-Temporal 1
    x = TemporalConvNet(filters=32, kernel_size=2, dilation_rate=1)(x)
    x = AdaptiveGraphConv(A_matrix, embed_dim=10)(x)
    
    # Layer Spatio-Temporal 2 (Peningkatan Receptive Field)
    x = TemporalConvNet(filters=64, kernel_size=2, dilation_rate=2)(x)
    x = AdaptiveGraphConv(A_matrix, embed_dim=10)(x)
    x = Dropout(0.2)(x)
    
    # BUG FIX: Jangan di-mean over time! Ambil timestep TERAKHIR karena konvolusi kita causal.
    # Ini memastikan model sangat responsif terhadap lonjakan akut di akhir sequence.
    x = x[:, :, -1, :] # Shape: (Batch, Nodes, Features)
    
    # Proyeksi ke target (TMA)
    outputs = Dense(1)(x) 
    outputs = Reshape((NUM_STATIONS,))(outputs)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.002), 
                  loss=asymmetric_mse, metrics=[keras.metrics.RootMeanSquaredError(name='rmse')])
    return model

# Verifikasi kompilasi model
test_model = build_gwnet(A_norm)
test_model.summary()
'''),

    new_markdown_cell('## 7. Pelatihan K-Fold Cross Validation\nPelatihan yang kokoh memerlukan validasi silang (K-Fold), dikombinasikan dengan penghentian dini (*Early Stopping*) untuk mencegah *overfitting*.'),
    new_code_cell('''kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_models = []
fold_histories = []

print("Memulai Pelatihan K-Fold Cross Validation...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full)):
    print(f"\\n--- Menginisiasi Fold {fold + 1} ---")
    
    X_train, y_train = X_train_full[train_idx], y_train_scaled[train_idx]
    X_val, y_val = X_train_full[val_idx], y_train_scaled[val_idx]
    
    model = build_gwnet(A_norm)
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ModelCheckpoint(output_path + f'gwnet_best_fold_{fold+1}.keras', monitor='val_rmse', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=0
    )
    
    best_rmse = min(history.history['val_rmse'])
    print(f"Fold {fold + 1} Selesai. Best Validation RMSE (Scaled): {best_rmse:.4f}")
    
    fold_models.append(model)
    fold_histories.append(history)
'''),

    new_markdown_cell('## 8. Evaluasi Model dan Visualisasi\nAnalisis konvergensi fungsi objektif (Loss) dan pengujian rekonstruksi sinyal air pada data validasi independen.'),
    new_code_cell('''fig, ax = plt.subplots(1, 2, figsize=(18, 6))

# ==========================================
# 1. Permutation Feature Importance (GWNet)
# ==========================================
print("Menghitung Permutation Feature Importance...")
baseline_preds = fold_models[-1].predict(X_val, verbose=0)
baseline_rmse = root_mean_squared_error(y_val.flatten(), baseline_preds.flatten())

feature_importances = []
for f_idx, feat in enumerate(env_features):
    X_val_shuffled = X_val.copy()
    
    # Mengacak fitur ke-f_idx untuk merusak informasinya
    shape = X_val_shuffled[:, :, :, f_idx].shape
    flat = X_val_shuffled[:, :, :, f_idx].flatten()
    np.random.shuffle(flat)
    X_val_shuffled[:, :, :, f_idx] = flat.reshape(shape)
    
    shuffled_preds = fold_models[-1].predict(X_val_shuffled, verbose=0)
    shuff_rmse = root_mean_squared_error(y_val.flatten(), shuffled_preds.flatten())
    
    # Importance = Kenaikan Error (semakin tinggi = semakin penting fiturnya)
    importance = shuff_rmse - baseline_rmse
    feature_importances.append((feat, importance))

feature_importances.sort(key=lambda x: x[1], reverse=True)
feats = [x[0] for x in feature_importances][:20] # Top 20
imps = [x[1] for x in feature_importances][:20]

sns.barplot(x=imps, y=feats, ax=ax[0], palette='viridis')
ax[0].set_title('Top Feature Importance (Permutation pada GWNet)')
ax[0].set_xlabel('Kenaikan RMSE (Jika Fitur Diacak)')

# ==========================================
# 2. Actual vs Predicted (Validasi Fold Terakhir)
# ==========================================
st_index = 0 # Stasiun Pertama (bisa diganti stasiun lain)
sub_actual_scaled = y_val[-200:] 
sub_actual = sub_actual_scaled * target_std + target_mean

val_preds_scaled = fold_models[-1].predict(X_val[-200:], verbose=0)
val_preds = val_preds_scaled * target_std + target_mean

ax[1].plot(sub_actual[:, st_index], label='Actual TMA (MDPL)', color='blue', alpha=0.7)
ax[1].plot(val_preds[:, st_index], label='Predicted TMA', color='red', alpha=0.7)
ax[1].set_title(f'Actual vs Predicted (Validation Set - {station_names[st_index]})')
ax[1].legend()

plt.tight_layout()
plt.show()
'''),

    new_markdown_cell('## 9. Inferensi dan Pembentukan Berkas Submission\nMelakukan prediksi ansambel menggunakan rata-rata keluaran seluruh model Fold, kemudian memformatnya sesuai spesifikasi kompetisi Kaggle.'),
    new_code_cell('''print("Melakukan inferensi massal ansambel (Ensemble Inference)...")

ensemble_preds_scaled = np.zeros((len(X_test_full), NUM_STATIONS), dtype=np.float32)
for m in fold_models:
    ensemble_preds_scaled += m.predict(X_test_full, verbose=0)
ensemble_preds_scaled /= len(fold_models)

# Reversi normalisasi data
test_preds = ensemble_preds_scaled * target_std + target_mean

# Format ke luaran Kaggle
submission_rows = []
for i, t in enumerate(test_valid_times):
    for j, station in enumerate(station_names):
        id_str = f"{t.strftime('%Y-%m-%d %H:%M:%S')} - {station}"
        tma_val = test_preds[i, j]
        submission_rows.append({'id': id_str, 'tma_mdpl': tma_val})
        
sub_df = pd.DataFrame(submission_rows)

sub_final = pd.merge(sample_sub[['id']], sub_df, on='id', how='left')
sub_final['tma_mdpl'] = sub_final['tma_mdpl'].fillna(method='ffill').fillna(method='bfill')

display(sub_final.head(10))
sub_path = output_path + 'submission_v9_gwnet.csv'
sub_final.to_csv(sub_path, index=False)
print(f"\\nProses inferensi selesai. Berkas '{sub_path}' berhasil diekspor.")
''')
])

with open('c:/Users/LENOVO X1 CARBON/Documents/SSF-SSDS/V9_GWNET_TMA.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
