import pandas as pd
import numpy as np
import folium
from folium import plugins

def generate_river_map():
    print("Membaca data untuk merekonstruksi topologi sungai...")
    path = './'
    train_df = pd.read_csv(path + 'train.csv')
    koordinat_df = pd.read_csv(path + 'data_pendukung/koordinat_pos.csv')

    # 1. Ekstrak Elevasi (Rata-rata mdpl)
    # Air mengalir dari mdpl tinggi (hulu) ke mdpl rendah (hilir)
    elevasi = train_df.groupby('nama_pos')['tma_mdpl'].mean().reset_index()
    elevasi.columns = ['nama_pos', 'elevasi_mdpl']

    # Gabung dengan koordinat
    df_map = pd.merge(koordinat_df, elevasi, on='nama_pos', how='left')
    df_map = df_map.dropna(subset=['latitude', 'longitude', 'elevasi_mdpl']).reset_index(drop=True)

    # 2. Algoritma Heuristik Aliran Sungai 
    # Aturan: Air mengalir ke titik terdekat yang elevasinya lebih rendah
    coords = df_map[['latitude', 'longitude']].values
    elevations = df_map['elevasi_mdpl'].values
    
    flow_edges = []
    for i in range(len(df_map)):
        # Cari semua pos yang elevasinya lebih rendah
        lower_idx = np.where(elevations < elevations[i])[0]
        
        if len(lower_idx) > 0:
            # Hitung jarak dari pos i ke semua pos yang lebih rendah menggunakan NumPy
            dist = np.sqrt(((coords[lower_idx] - coords[i]) ** 2).sum(axis=1))
            
            # Cari yang jarak geografisnya paling dekat
            nearest_lower_idx = lower_idx[np.argmin(dist)]
            min_dist = np.min(dist)
            
            # Batasi koneksi maksimal ~50km (0.5 derajat) agar tidak menyeberang sistem sungai lain (seperti Pacitan)
            if min_dist < 0.6: 
                flow_edges.append((i, nearest_lower_idx))

    # 3. Menggambar Peta dengan Folium
    print("Menggambar peta dengan aliran AntPath...")
    m = folium.Map(location=[-7.4, 111.5], zoom_start=9, tiles='CartoDB positron')

    # Color map untuk menandakan Hulu (Merah/Tinggi) ke Hilir (Biru/Rendah)
    colormap = folium.LinearColormap(
        colors=['#0000FF', '#00FFFF', '#FFFF00', '#FF0000'], 
        vmin=df_map['elevasi_mdpl'].min(),
        vmax=df_map['elevasi_mdpl'].max(),
        caption='Elevasi (Meter di atas permukaan laut)'
    )
    m.add_child(colormap)

    # Gambar Aliran Sungai (Animasi Semut Berjalan)
    for src, dst in flow_edges:
        src_coord = coords[src]
        dst_coord = coords[dst]
        
        plugins.AntPath(
            locations=[src_coord, dst_coord],
            dash_array=[10, 20],
            delay=1000, # Kecepatan aliran
            color='blue',
            pulse_color='cyan',
            weight=4,
            opacity=0.6,
            tooltip=f"Aliran dari {df_map.iloc[src]['nama_pos']} ke {df_map.iloc[dst]['nama_pos']}"
        ).add_to(m)

    # Gambar Titik Stasiun
    for i in range(len(df_map)):
        row = df_map.iloc[i]
        color = colormap(row['elevasi_mdpl'])
        
        # 🔥 HIGHLIGHT KHUSUS BOJONEGORO 🔥
        if 'Bojonegoro' in row['nama_pos']:
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=f"<div style='width:200px'><b>🚨 {row['nama_pos']} 🚨</b><br>Elevasi: <b>{row['elevasi_mdpl']:.2f} mdpl</b><br><br>Ini adalah titik temu aliran dari pos-pos hulu sebelum menuju laut. Elevasinya sangat rendah, sehingga rentan fluktuasi pasang surut laut dan penumpukan debit dari hulu!</div>",
                tooltip=f"TARGET: {row['nama_pos']}",
                icon=folium.Icon(color='red', icon='star', prefix='fa')
            ).add_to(m)
            
            # Radius bahaya
            folium.Circle(
                location=[row['latitude'], row['longitude']],
                radius=15000,
                color='red',
                fill=True,
                fill_opacity=0.2
            ).add_to(m)
        else:
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=8,
                popup=f"<b>{row['nama_pos']}</b><br>Elevasi: {row['elevasi_mdpl']:.2f} mdpl",
                tooltip=row['nama_pos'],
                color='black',
                weight=1,
                fill=True,
                fill_color=color,
                fill_opacity=0.9
            ).add_to(m)

    # 4. Simpan Peta
    file_name = 'Peta_Aliran_Bengawan_Solo.html'
    m.save(file_name)
    print(f"✅ Selesai! Buka file '{file_name}' di browser Anda.")

if __name__ == '__main__':
    generate_river_map()
