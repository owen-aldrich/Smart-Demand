# Smart-Demand: Sales Volume Prediction System

Random Forest machine learning model trained on the Brazilian E-Commerce (Olist) dataset. Predicts total sales volume per product category per month based on pricing, quality, and market factors.

COMP6577001 - Machine Learning | Final Project

## Struktur Folder

```
SmartDemand_App/
├── main.py
├── index.html
├── requirements.txt
└── models/
    ├── random_forest_model.joblib
    ├── linear_regression_model.joblib
    ├── label_encoder.joblib
    └── config.json
```

## Run Lokal

```
pip install -r requirements.txt
python main.py
```

Buka browser di http://127.0.0.1:8000

## Deployment

Saat ini aplikasi belum memiliki demo yang di-hosting. Repository ini sudah mencakup kode, model, dan konfigurasi yang diperlukan untuk di-deploy ulang ke platform seperti Railway atau Render.

## Model Performance

| Model | Test R² | Test MAE | Test RMSE |
|---|---|---|---|
| Random Forest | 0.9541 | 8.92 | 17.22 |
| Linear Regression | 0.9445 | 9.45 | 18.26 |

## Notebooks

Repository ini menyertakan dua notebook yang menunjukkan proses iterasi pengembangan model.

**01_Initial_Model_ProductLevel.ipynb**

Pendekatan awal, target prediksi adalah quantity_sold per produk individual per bulan. Hasilnya R² hanya 0.5033 karena data sangat noisy, mayoritas produk hanya terjual 1-5 unit per bulan sehingga sulit diprediksi dari fitur yang tersedia.

**02_Final_Model_CategoryLevel.ipynb**

Setelah evaluasi, granularitas data diubah menjadi per kategori produk per bulan, bukan per produk individual. Pendekatan ini menghasilkan R² 0.9541 karena data jauh lebih smooth dan konsisten. Model ini yang digunakan pada aplikasi yang di-deploy.

## Tech Stack

- Backend: FastAPI (Python)
- Frontend: HTML / CSS / Vanilla JS
- Model: scikit-learn (Random Forest Regressor)
