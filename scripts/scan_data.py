import pandas as pd
import numpy as np

print("=== SCANNING UNIVERSE DATA ===")
tr = pd.read_csv('train.csv')
te = pd.read_csv('test.csv')
env = pd.read_csv('data_pendukung/data_lingkungan.csv')
coord = pd.read_csv('data_pendukung/koordinat_pos.csv')

print(f"Train size: {tr.shape}, Test size: {te.shape}")
print(f"Stations count: {tr['nama_pos'].nunique()}")

print("\\n1. TARGET DISTRIBUTION (TMA_MDPL):")
print(tr['tma_mdpl'].describe())

print("\\n2. MISSING VALUES:")
print(tr.isna().sum())
print(env.isna().sum())

print("\\n3. CORRELATION WITH TARGET (Top 5 features):")
# Merge to check correlation
merged = pd.merge(tr, env, on=['datetime', 'nama_pos'], how='inner')
numeric_cols = merged.select_dtypes(include=[np.number]).columns
corr = merged[numeric_cols].corr()['tma_mdpl'].sort_values(ascending=False)
print(corr.head(10))
