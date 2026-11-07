# 🌊 Bengawan Solo Flood Forecasting: V36 End-to-End Architecture Design
## The Normalized Global Ensemble

---

### 1. Definisi Kompetisi & Objektif (The Task)
Kompetisi ini menantang peserta untuk memprediksi Tinggi Muka Air atau TMA (`tma_mdpl`) pada 30 stasiun pengamatan yang membentang di sepanjang Daerah Aliran Sungai (DAS) Bengawan Solo. 
*   **Target Periode:** Memprediksi fluktuasi TMA di masa depan (mulai 19 September 2025 hingga Mei 2026).
*   **Metrik Evaluasi:** Root Mean Squared Error (RMSE). Semakin kecil *error* antara tebakan dan kenyataan, semakin tinggi peringkat di *Leaderboard* Kaggle.

Tantangan utamanya adalah **Heterogenitas Spasial**. Setiap stasiun memiliki elevasi dasar, lebar sungai, dan perilaku topologi yang sangat berbeda (misalnya: Babat bisa mengalami lonjakan 10 meter saat banjir, sedangkan Jurug mungkin maksimal hanya 2 meter). 

---

### 2. Penjelasan Dataset
Proyek ini ditenagai oleh kombinasi dataset utama dan data pendukung (spasial-klimatologi) untuk membangun sistem kecerdasan buatan hidrologi yang utuh.

#### A. Dataset Utama (Time-Series TMA)
*   **`train.csv`**: Kumpulan data historis yang berisi ~27.000 baris observasi (sekitar 900 observasi observasi harian/jam per stasiun). Memiliki kolom `id`, `datetime` (waktu observasi), `nama_pos` (nama 30 stasiun pengamatan), dan `tma_mdpl` (elevasi muka air). Data ini berakhir pada 18 September 2025.
*   **`test.csv`**: Dataset inferensi yang mewajibkan kita menebak nilai `tma_mdpl` untuk periode waktu setelah 18 September 2025 di 30 stasiun yang sama.
*   **`sample_submission.csv`**: Format standar Kaggle untuk pengumpulan (*submission*).

#### B. Data Pendukung (*Supporting Data*)
*   **`data_lingkungan.csv`**: Dataset vital berbasis iklim historis (berasal dari Open-Meteo) yang sejajar waktunya dengan dataset utama. Berisi fitur-fitur cuaca ekstrem, termasuk:
    *   Curah hujan (`rainfall_mm`), suhu (`temperature_c`), kelembaban (`humidity_pct`).
    *   Kelembaban tanah pada berbagai kedalaman (`soil_moisture_0_7cm`, `7_28cm`, `28_100cm`, `100_255cm`).
    *   Indikator iklim makro global seperti anomali laut MJO (`mjo_amplitude`) dan El-Niño (`nino_34`).
*   **`koordinat_pos.csv`**: Berisi letak astronomis (*latitude, longitude*) dari ke-30 stasiun. Sangat penting untuk menghitung jarak fisik antar stasiun.
*   **Database Eksternal (HydroRIVERS v10)**: Referensi injeksi manual yang memberikan 3 fitur topologi bawaan (*Static Features*) untuk setiap stasiun, yakni Luas Tangkapan Air (`UPLAND_SKM`), Debit Rata-rata (`DIS_AV_CMS`), dan Ordo Sungai (`ORD_STRA`).

---

### 3. Latar Belakang & Analisis Kegagalan Terdahulu
Dalam upaya mencapai RMSE < 0.81, kita telah melalui berbagai eksperimen. Hasil observasi dari *Leaderboard* memberikan kesimpulan krusial:

*   **V26 (30 Model Lokal Sederhana):** Skor Validasi K-Fold sangat baik (0.98), namun kinerjanya sangat rawan terhadap anomali karena fiturnya terbatas. Tiap model hanya belajar dari 900 baris sehingga rawan *overfitting* jika fitur diperbanyak.
*   **V34 (Model Global dengan Target Asli):** Model gagal menangkap "puncak ekstrem" di stasiun tertentu (seperti Babat) karena algoritma global mencari *loss* rata-rata (*mean*) dari seluruh 27.000 baris dan takut menebak angka ekstrem yang bisa merusak akurasi di stasiun ber-elevasi rendah (seperti Jurug).
*   **V35 (30 Model Lokal + 50 Fitur Kompleks):** Menghasilkan skor LB **1.93** (*Massive Overfitting*). Memasukkan 50 variabel kompleks ke dalam porsi 900 baris data per stasiun menyebabkan model "menghafal mati" data masa lalu secara brutal dan berhalusinasi saat melihat masa depan.
*   **Arsitektur Kompetitor (Global Z-Score + 84 Fitur):** Mencapai skor LB **1.60**. Meskipun menggunakan fitur yang jauh lebih banyak dari V35, mereka terhindar dari *overfitting* karena menggunakan kekuatan penuh **Model Global (27.000 baris data komposit)**.

**Kesimpulan Absolut:** 
Untuk dapat menggunakan 50+ fitur hidrologi tanpa *overfitting*, kita **wajib menggunakan Model Global**. Namun, agar Model Global memiliki insting menangkap banjir ekstrem secara spesifik per stasiun (seperti V26), kita **wajib memanipulasi targetnya menggunakan Normalisasi Z-Score Spasial**.

---

### 4. Konsep Utama V36: Target-Normalized Global Architecture

Arsitektur V36 dirancang menggabungkan keunggulan data komposit (Anti-Overfit) dengan keunikan topologi tiap stasiun (Anti-Underfit).

#### A. Z-Score Robust Target Normalization (Per Station)
Alih-alih memprediksi Tinggi Muka Air dalam satuan "Meter" secara langsung, V36 akan memprediksi **Z-Score (Skor Deviasi)**.
Pada tahap *preprocessing*, `tma_mdpl` tiap stasiun dinormalisasi menggunakan statistik yang kebal terhadap *outlier* (Robust Scaler):

$$ Z(pos, t) = \frac{TMA(pos, t) - Median(pos)}{\max(IQR(pos) / 1.349, \ 0.05)} $$

*   **Median(pos)** = Nilai tengah elevasi historis di stasiun tersebut.
*   **IQR(pos)** = Interquartile Range historis di stasiun tersebut.

**Dampak Matematis:**
Jika Babat memiliki elevasi ekstrem 10 meter dan Jurug memiliki elevasi ekstrem 2 meter, dengan rumusan di atas keduanya kini akan direpresentasikan sebagai **+4 Z-Score**. 
Model Global kini bisa mempelajari *physics engine* cuaca secara universal: *"Jika hujan kumulatif 72 jam tinggi dan tanah jenuh air, maka sungai akan meluap sebesar +4 Z-Score"*. Ia tidak lagi takut menebak angka ekstrem!

#### B. Fitur Spasial-Temporal (The 50 Features)
V36 mengandalkan rekayasa fitur (*Feature Engineering*) berbasis sains hidrologi:
1.  **Antecedent Precipitation Index (API EWM):** Serapan curah hujan historis (tanah basah) menggunakan *Exponential Weighted Moving Average* (halflife 12, 48, 168 jam).
2.  **Spatial Upstream Rain:** Perhitungan curah hujan kiriman (hulu) menggunakan matriks bobot eksponensial berdasarkan jarak fisik (Haversine) antar stasiun.
3.  **HydroRIVERS Features:** Fitur statis (Luas DAS, Order Sungai, Debit Rata-rata) yang membuat model Global bisa membedakan karakter fisik kapasitas tiap stasiun.
4.  **Target Lag (Memory):** Elevasi TMA persis 365 hari (1 tahun) sebelumnya untuk menangkap memori siklus musiman tahunan.
5.  **Effective Runoff:** Rasio perkalian antara curah hujan langsung dengan tingkat saturasi tanah saat ini.

#### C. Algoritma (The Grand Ensemble)
Karena target telah diubah menjadi Z-Score (yang mengandung nilai negatif), objektif `Tweedie` tidak relevan. V36 menggunakan **RMSE** atau **Huber Loss**.
Mesin prediksi terdiri dari paduan 3 algoritma *State-of-the-Art* (*Grand Ensemble*):
*   **XGBoost Regressor**
*   **LightGBM Regressor**
*   **CatBoost Regressor** (Sangat handal untuk `nama_pos` sebagai *categorical feature*).

#### D. Inverse Transformation (Post-Processing)
Setelah ketiga model selesai menebak Z-Score untuk data Test Set, angka tersebut wajib dibongkar kembali (*Inverse Transform*) menjadi "Meter Asli" untuk direkam ke `submission_v36.csv`:

$$ TMA_{pred}(pos, t) = \left(Z_{pred}(pos, t) \times \max(IQR(pos) / 1.349, \ 0.05)\right) + Median(pos) $$

---

### 5. Alur Eksekusi
1.  **Pra-pemrosesan Data:** Menghitung dan menyimpan kamus Median & IQR untuk seluruh 30 stasiun. Membangun 50 fitur sains lingkungan dari dataset pendukung.
2.  **Pembelajaran Global:** Melatih XGBoost, LightGBM, dan CatBoost secara serentak pada gabungan 27.000 data observasi dengan target Z-Score.
3.  **Inferensi & Pembalikan:** Memprediksi *Test Set*, merata-ratakan skor ketiga model, lalu membalikkan skala Z-Score kembali ke batas rasional Meter Asli.
4.  **Validasi Visual:** Mengekstrak *Fitted Train Predictions* untuk membandingkan secara visual (lewat grafik *line chart*) sejauh mana V36 sukses menusuk puncak ekstrem banjir historis layaknya V26.
