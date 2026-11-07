import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import box

print("1. Membaca file Shapefile HydroRIVERS...")
# Membaca shapefile (Geopandas otomatis membaca .dbf, .shx, .prj yang ada di folder yang sama)
shp_path = "c:/Users/LENOVO X1 CARBON/Documents/SSF-SSDS/data_pendukung/HydroRIVERS_v10_au_shp/HydroRIVERS_v10_au.shp"
rivers_au = gpd.read_file(shp_path)

print(f"Total segmen sungai se-Australasia: {len(rivers_au)}")

print("\n2. Melakukan Cropping (Spatial Filter) ke area Bengawan Solo...")
# Koordinat Bounding Box Bengawan Solo (Perkiraan Kasar)
# Min Lon: 110.5, Min Lat: -8.3 | Max Lon: 112.8, Max Lat: -6.8
bbox = box(110.5, -8.3, 112.8, -6.8)

# Menggunakan sindex (Spatial Index) agar filter jauh lebih cepat
rivers_bs = rivers_au[rivers_au.geometry.intersects(bbox)]

print(f"Total segmen sungai di sekitar Bengawan Solo: {len(rivers_bs)}")

print("\n3. Mengekstrak Atribut Geospasial Kunci:")
# Menampilkan beberapa kolom penting dari database atribut (.dbf)
columns_of_interest = ['HYRIV_ID', 'NEXT_DOWN', 'MAIN_RIV', 'LENGTH_KM', 'CATCH_SKM', 'UPLAND_SKM', 'DIS_AV_CMS', 'ORD_STRA']
print(rivers_bs[columns_of_interest].head())

print("\n4. Membuat Visualisasi Peta Jaringan Sungai...")
fig, ax = plt.subplots(figsize=(12, 8))

# Mewarnai dan menebalkan sungai berdasarkan Order Strahler (ORD_STRA)
# Order 1 (anak sungai kecil) -> tipis
# Order 6+ (sungai utama) -> tebal
for str_order in range(1, 9):
    subset = rivers_bs[rivers_bs['ORD_STRA'] == str_order]
    if len(subset) > 0:
        subset.plot(ax=ax, linewidth=str_order*0.5, color='blue', alpha=0.7)

# Overlay dengan stasiun kita jika ada koordinat_pos.csv
try:
    coords = pd.read_csv("c:/Users/LENOVO X1 CARBON/Documents/SSF-SSDS/data_pendukung/koordinat_pos.csv")
    coords_gdf = gpd.GeoDataFrame(
        coords, 
        geometry=gpd.points_from_xy(coords.longitude, coords.latitude),
        crs="EPSG:4326"
    )
    coords_gdf.plot(ax=ax, color='red', marker='^', markersize=50, label='Stasiun Pengamat')
    
    # Menambah nama stasiun
    for x, y, label in zip(coords_gdf.geometry.x, coords_gdf.geometry.y, coords_gdf['nama_pos']):
        ax.text(x, y, label, fontsize=8, ha='right', color='darkred')
except Exception as e:
    print(f"Gagal meload stasiun: {e}")

ax.set_title("Peta Jaringan Sungai HydroRIVERS - Bounding Box Bengawan Solo")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.legend()
plt.tight_layout()
plt.savefig("c:/Users/LENOVO X1 CARBON/Documents/SSF-SSDS/HydroRIVERS_Bengawan_Solo_Map.png", dpi=300)
print("Peta berhasil disimpan sebagai 'HydroRIVERS_Bengawan_Solo_Map.png'")
