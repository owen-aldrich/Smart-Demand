# Smart-Demand: Category-Level Order Forecasting

Aplikasi FastAPI dengan Random Forest dan Linear Regression dari notebook final ketiga, `03_Final_Model.ipynb`. Target adalah jumlah item pesanan per kategori per bulan pembelian, termasuk item dari order yang kemudian dibatalkan.

COMP6577001 — Machine Learning | Final Project

## Run Lokal

Gunakan Python 3.10 atau lebih baru. Buka PowerShell atau terminal VS Code di folder proyek. Untuk lokasi proyek saat ini di Windows, jalankan:

```powershell
cd D:\SmartDemand-main
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

Jika folder proyek dipindahkan, sesuaikan perintah `cd`. Perintah di atas menggunakan Python dari virtual environment secara langsung sehingga tidak memerlukan aktivasi PowerShell.

Setelah terminal menampilkan server berjalan, buka **http://127.0.0.1:8000** di browser. Biarkan terminal tetap berjalan selama aplikasi digunakan. Buka halaman melalui alamat tersebut; membuka `index.html` dengan klik dua kali (`file://`) tidak menghubungkan halaman ke API prediksi.

Untuk pemakaian berikutnya, cukup jalankan:

```powershell
cd D:\SmartDemand-main
.\.venv\Scripts\python.exe main.py
```

Hentikan server dengan **Ctrl+C** di terminal. Jika model atau `config.json` diganti dengan hasil training baru, hentikan dan jalankan ulang server, lalu muat ulang halaman browser.

Runtime scikit-learn dipasang pada versi **1.6.1**, sesuai artefak training. Model hasil training sudah tersedia di folder `models/`, sehingga notebook tidak perlu dijalankan ulang hanya untuk memakai aplikasi. Aplikasi belum dipublikasikan ke hosting.

### Jika Aplikasi Tidak Terbuka

- Jika `python` tidak dikenali, pastikan Python terpasang dan tersedia di PATH. Jika tersedia Python Launcher, gunakan `py -3 -m venv .venv` untuk membuat virtual environment.
- Jika muncul `ModuleNotFoundError`, jalankan ulang perintah instalasi dengan `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.
- Jika browser menampilkan koneksi ditolak, pastikan proses server masih berjalan dan tidak menampilkan error di terminal.
- Jika port 8000 sedang digunakan, jalankan `.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8001`, lalu buka http://127.0.0.1:8001.

## Artefak Model

```text
models/
├── random_forest_model.joblib
├── linear_regression_model.joblib
├── label_encoder.joblib
└── config.json
```

Kedua file model berisi sklearn Pipeline lengkap: imputasi, scaling, OneHotEncoder, dan estimator. `label_encoder.joblib` adalah ekspor **OneHotEncoder** yang sudah di-fit pada training; nama file mengikuti struktur proyek lama. Jangan melakukan encoding kategori lagi sebelum memanggil pipeline.

`config.json` menyimpan urutan fitur, kategori training, lookup seasonality dari training, parameter RF, periode train/test, serta metrik aktual. Semua artefak harus berasal dari run notebook yang sama.

## Input Prediksi

- Pilih kategori dan bulan target.
- Isi jumlah item pesanan pada tiga bulan kalender sebelumnya, dari yang paling lama ke paling baru. API menghitung `sales_avg_3m` dari ketiganya.
- Isi harga rata-rata, ongkir rata-rata, rating yang sudah tersedia, jumlah produk unik terjual, dan rentang harga dari **bulan sebelumnya**.
- Harga dan ongkir memakai **BRL**, sama dengan dataset Olist. Tidak ada konversi Rupiah dengan kurs asumsi.
- Rentang harga adalah harga item maksimum dikurangi minimum, bukan perkiraan dari harga rata-rata. Jumlah produk tidak lagi diisi konstan oleh aplikasi.
- Rating hanya mencakup review yang dijawab sebelum bulan target. Kosongkan kondisi numerik yang tidak tersedia agar memakai imputer training.
- Jika tidak ada pesanan bulan lalu, isi jumlah produk unik nol dan kosongkan harga, ongkir, rating, serta rentang harga.

Diskon tidak menjadi fitur model ini. Target bulan prediksi tidak boleh dimasukkan ke histori.

## API

`GET /api/config` menyediakan kategori, skema fitur, seasonality, dan metrik test.

`POST /api/predict` menerima input seperti berikut. Contoh ini memprediksi Mei dari histori Februari–April:

```json
{
  "category": "electronics",
  "month": 5,
  "sales_history": [80, 90, 100],
  "prev_avg_price": 100,
  "prev_avg_freight": 20,
  "prev_avg_rating": 4,
  "prev_num_products": 50,
  "prev_price_range": 150,
  "model": "rf"
}
```

Gunakan `"lr"` untuk Linear Regression. API mengirim DataFrame dengan urutan fitur persis seperti notebook ke pipeline, lalu memotong prediksi negatif ke nol. `raw_units` adalah hasil tanpa pembulatan; `units` dibulatkan untuk tampilan.

Respons juga menampilkan prediksi baseline bulan lalu dan rata-rata tiga bulan. Untuk RF, tree forecast spread menunjukkan sebaran prediksi pohon, **bukan probabilitas terkalibrasi**. Linear Regression tidak diberi distribusi probabilitas buatan.

Volume band dan gauge adalah kategori/indeks tampilan tetap, bukan keluaran classifier, akurasi, atau confidence. Batas 70/120/180 dan referensi indeks 300 unit tidak dipelajari dari data dan tidak disesuaikan per kategori.

## Hasil Evaluasi (`test_result`)

Train: Desember 2016–Maret 2018, 901 baris. Test: April–Agustus 2018, 366 baris. Split berdasarkan bulan; parameter RF tetap seperti notebook 02, tanpa validation atau tuning. Model utama RF ditetapkan sebelum test dan tidak otomatis diganti berdasarkan skor test.

Tabel berikut mengikuti output cell `test_result` notebook 03 dan metrik pada `models/config.json`. MAE dan RMSE memakai satuan item pesanan per kategori per bulan; semakin kecil semakin baik. Untuk R², semakin besar semakin baik.

| Model | MAE (unit) | RMSE (unit) | R² |
|---|---:|---:|---:|
| Last Month Baseline | 16.7787 | 33.2357 | 0.9640 |
| 3-Month Average Baseline | 20.9499 | 43.9283 | 0.9371 |
| Linear Regression | 23.3814 | 45.6279 | 0.9321 |
| Random Forest | 27.4954 | 58.9179 | 0.8868 |

Pada periode ini, **Last Month Baseline memiliki hasil terbaik** pada ketiga metrik. Linear Regression mengungguli Random Forest, tetapi keduanya belum mengalahkan baseline. R² bukan persentase prediksi yang tepat. Aplikasi mempertahankan pilihan RF/LR untuk proyek, tanpa mengklaim RF memberi perbaikan dibanding baseline.

Evaluasi notebook ditampilkan sebagai tabel di output cell, tanpa mengekspor CSV evaluasi, laporan prediksi per baris, atau forecast tambahan. Metrik tetap disimpan dalam `config.json` untuk ditampilkan aplikasi.

Model mengasumsikan histori sumber lengkap dan informasi transaksi tersedia pada timestamp pembelian. Rating dibatasi menurut waktu jawab review. Kategori dengan kurang dari tiga bulan histori tidak tercakup evaluasi.

## Notebooks

- `01_Initial_Model_ProductLevel.ipynb`: eksperimen awal tingkat produk.
- `02_Final_Model_CategoryLevel.ipynb`: eksperimen kategori lama dengan random split dan fitur yang menyebabkan leakage; skor lamanya tidak menjadi metrik aplikasi saat ini.
- `03_Final_Model.ipynb`: notebook final ketiga dengan pipeline mandiri yang sesuai dengan skema artefak aktif. Memilih kolom sebelum join, menggunakan histori tiga bulan, rating yang tersedia sebelum cutoff, seasonality training saja, dan split waktu. Menampilkan tabel `test_result` saat dijalankan, lalu menyimpan tiga `.joblib` (dua model dan satu encoder) beserta `config.json` ke folder `models/`.

## Pengujian

Untuk pengujian API dan notebook:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Smoke test browser memakai Chrome/Edge headless dan Node 22+ tanpa paket npm:

```powershell
$env:SMARTDEMAND_PYTHON = (Resolve-Path .\.venv\Scripts\python.exe).Path
node tests/browser_smoke.cjs
```

Tes mencakup kecocokan prediksi API dengan pipeline asli, imputasi rating kosong, validasi input, penyimpanan encoder/model, pemeriksaan temporal, serta alur RF/LR di browser.
