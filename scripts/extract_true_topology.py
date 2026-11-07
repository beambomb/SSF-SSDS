import geopandas as gpd
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

print("1. Memuat Data Koordinat Stasiun...")
coords = pd.read_csv('data_pendukung/koordinat_pos.csv')
gdf_coords = gpd.GeoDataFrame(
    coords, 
    geometry=gpd.points_from_xy(coords.longitude, coords.latitude), 
    crs="EPSG:4326"
)

print("2. Memuat Shapefile HydroRIVERS (Proses ini mungkin memakan waktu ~10 detik)...")
rivers = gpd.read_file('data_pendukung/HydroRIVERS_v10_au_shp/HydroRIVERS_v10_au.shp')
# Crop ke area Bengawan Solo agar komputasi lebih cepat
rivers_bs = rivers.cx[110.5:113.0, -8.5:-6.5].copy()

print("3. Melakukan Spatial Join (Mencocokkan titik stasiun dengan garis sungai)...")
station_hyriv = {}
for idx, row in gdf_coords.iterrows():
    # Hitung jarak dari stasiun ke semua garis sungai di area BS
    distances = rivers_bs.geometry.distance(row.geometry)
    nearest_idx = distances.idxmin()
    hyriv_id = rivers_bs.loc[nearest_idx, 'HYRIV_ID']
    station_hyriv[row['nama_pos']] = hyriv_id
    print(f"   [Mapped] {row['nama_pos']} -> HYRIV_ID: {hyriv_id}")

print("\\n4. Membangun Silsilah Jaringan (NEXT_DOWN)...")
next_down_map = rivers_bs.set_index('HYRIV_ID')['NEXT_DOWN'].to_dict()

# upstream_map[HILIR] = [HULU_1, HULU_2, ...]
upstream_map = {pos: [] for pos in coords['nama_pos']}

print("5. Menelusuri Aliran Air (Tracing)...")
for pos, start_hyriv in station_hyriv.items():
    current_hyriv = start_hyriv
    visited = set()
    
    # Telusuri terus ke bawah (Hilir) sampai ke laut (0) atau keluar peta
    while current_hyriv in next_down_map and current_hyriv != 0:
        visited.add(current_hyriv)
        current_hyriv = next_down_map[current_hyriv]
        
        # Apakah segmen sungai di bawah ini milik stasiun lain?
        for other_pos, other_hyriv in station_hyriv.items():
            if pos != other_pos and current_hyriv == other_hyriv:
                # 'pos' (Hulu) airnya sampai ke 'other_pos' (Hilir)
                # Maka bagi 'other_pos', 'pos' adalah stasiun Hulu-nya!
                if pos not in upstream_map[other_pos]:
                    upstream_map[other_pos].append(pos)
                    print(f"   🌊 Aliran Ditemukan: {pos} (Hulu) --> {other_pos} (Hilir)")
                    
        # Cegah infinite loop jika ada anomali data melingkar
        if current_hyriv in visited:
            break

print("\\n6. Menyimpan hasil ke 'true_upstream_map.json'...")
with open('true_upstream_map.json', 'w') as f:
    json.dump(upstream_map, f, indent=4)

print("✅ SELESAI! The True Topological Map berhasil diekstrak!")
