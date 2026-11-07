import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=== DEEP DIVE UNIVERSE SCAN ===")
tr = pd.read_csv('train.csv', parse_dates=['datetime'])
env = pd.read_csv('data_pendukung/data_lingkungan.csv', parse_dates=['datetime'])
merged = pd.merge(tr, env, on=['datetime', 'nama_pos'], how='inner')

print("\\n1. CROSS-CORRELATION (TMA vs RAINFALL LAGS) - Membuktikan Waktu Tempuh Air")
lag_corrs = {}
for lag in range(1, 15):
    merged_lag = merged.copy()
    merged_lag['rain_lag'] = merged_lag.groupby('nama_pos')['rainfall_mm'].shift(lag)
    corr = merged_lag[['tma_mdpl', 'rain_lag']].corr().iloc[0,1]
    lag_corrs[f"Lag {lag} Hari"] = corr

for k, v in lag_corrs.items():
    print(f"{k}: {v:.4f}")

print("\\n3. STATION VARIANCE ANALYSIS (Siapa yang paling bergejolak?)")
stats = tr.groupby('nama_pos')['tma_mdpl'].agg(['mean', 'std', 'max']).sort_values('std', ascending=False)
print(stats.head(5))

print("\\n4. MJO & EL NINO ANALYSIS")
print("Korelasi El Nino (nino_34):", merged['nino_34'].corr(merged['tma_mdpl']))
print("Korelasi MJO Amplitude:", merged['mjo_amplitude'].corr(merged['tma_mdpl']))
