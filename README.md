# LSTM TDS Tahmin Projesi

Sulama suyu TDS (tuzluluk) tahmini icin TensorFlow/Keras tabanli LSTM egitim pipeline'i.

## Proje Yapisi

```text
lstm_tds_project/
├── requirements.txt
├── README.md
├── data/
│   └── generate_sample.py
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   └── train.py
└── models/
```

## Gereksinimler

- Python 3.10.x (onerilen)
- CPU yeterli, GPU varsa TensorFlow otomatik kullanir.

## Kurulum

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

## Veri Hazirlama

USGS API (`parameterCd=00095`) ile veri cekmeyi dener. API hatasi/time-out olursa sentetik veri uretir.

```bash
python data/generate_sample.py
```

Varsayilan cikti: `data/tds_timeseries.csv`

## Egitim

```bash
python src/train.py
```

Hizli test (5 epoch):

```bash
python src/train.py --epochs 5
```

## Model ve Egitim Ayarlari

- Sequence length: `24`
- Split: `%70 train / %15 val / %15 test`
- Varsayilan feature kolonlari (4 adet):
  - `specific_conductance`
  - `temperature`
  - `weather_index`
  - `soil_type_code`
- Model:
  - `LSTM(50, return_sequences=True) + Dropout(0.2)`
  - `LSTM(50) + Dropout(0.2)`
  - `Dense(1)`
- Loss: `MSE`
- Optimizer: `Adam`
- Batch size: `32`
- Epoch: `50`
- Early stopping: `patience=10`, `monitor='val_loss'`
- Checkpoint: en iyi model kaydedilir

## Ciktilar

Egitim sonunda `models/` klasorunde:

- `tds_lstm_model.h5`
- `training_history.png`
- `predictions.png`
- `metrics.txt` (RMSE, MAE, R2)
- `metadata.json` (sequence length, feature kolonlari)
- `scalers.pkl` (normalizasyon scaler'lari)

## Tahmin Alma

Egitimden sonra sonraki saat(ler) icin tahmin:

```bash
python src/predict.py --steps 1
```

Varsayilan cikti: `models/forecast.csv`

Farkli ozellik seti ile egitmek istersen:

```bash
python src/train.py --feature-columns specific_conductance,temperature,weather_index,soil_type_code
```

## Hata Yonetimi

- USGS API timeout/HTTP hatasi: sentetik veriye otomatik fallback
- Eksik kolon/veri: yuklemede acik hata mesaji
- Eksik degerler: interpolation + bfill/ffill

## Not

USGS'den gelen `specific_conductance` degerinden TDS yaklasimi icin `TDS ~= 0.65 * SC` kullanilmistir.
