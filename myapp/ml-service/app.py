"""
Bu dosya MySQL'den veri okuyup LSTM modeli egiten ve tahmin ureten Flask ML servisidir.
"""

import os
import pickle
from datetime import datetime

import mysql.connector
import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential, load_model

load_dotenv()

app = Flask(__name__)

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "127.0.0.1"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", "esp32_user"),
    "password": os.getenv("DB_PASS", "esp32_pass"),
    "database": os.getenv("DB_NAME", "esp32_monitor"),
}

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.h5")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
WINDOW_SIZE = 10
MIN_TRAIN_ROWS = 50


def get_db_connection():
    """MySQL veritabanina baglanti olusturur."""
    return mysql.connector.connect(**DB_CONFIG)


def fetch_sensor_data(limit=None):
    """Sensor verilerini zaman sirasina gore alir."""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    query = "SELECT timestamp, salt, sicaklik FROM sensor_data ORDER BY timestamp ASC"
    params = []

    if limit is not None:
      query += " LIMIT %s"
      params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()

    cursor.close()
    conn.close()
    return rows


def build_sequences(values, window_size):
    """LSTM icin kaydirma pencereli giris/etiket dizileri hazirlar."""
    x_list = []
    y_list = []

    for i in range(window_size, len(values)):
        x_list.append(values[i - window_size:i])
        y_list.append(values[i])

    return np.array(x_list), np.array(y_list)


def build_model(input_shape):
    """Istenen mimariye uygun LSTM modelini olusturur."""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(2)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


@app.get("/health")
def health():
    """Servis durumunu doner."""
    return jsonify({"status": "ok", "service": "ml-service"})


@app.get("/train")
def train_model():
    """MySQL verisi ile modeli egitir ve model.h5 dosyasina kaydeder."""
    try:
        rows = fetch_sensor_data()
        if len(rows) < MIN_TRAIN_ROWS:
            return jsonify({
                "error": "Model egitimi icin en az 50 veri gerekli",
                "current_count": len(rows)
            }), 400

        raw_values = np.array([[float(r["salt"]), float(r["sicaklik"])] for r in rows], dtype=np.float32)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_values = scaler.fit_transform(raw_values)

        x_train, y_train = build_sequences(scaled_values, WINDOW_SIZE)

        if len(x_train) == 0:
            return jsonify({"error": "Yeterli sequence olusturulamadi"}), 400

        model = build_model((x_train.shape[1], x_train.shape[2]))
        model.fit(x_train, y_train, epochs=30, batch_size=16, verbose=0)

        model.save(MODEL_PATH)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)

        return jsonify({
            "message": "Model basariyla egitildi",
            "model_path": MODEL_PATH,
            "model_version": MODEL_VERSION,
            "trained_at": datetime.utcnow().isoformat() + "Z",
            "row_count": len(rows)
        })
    except Exception as exc:
        return jsonify({"error": "Egitim sirasinda hata", "details": str(exc)}), 500


@app.get("/predict")
def predict_next():
    """Son 10 veriyi kullanarak bir sonraki salt/sicaklik degerini tahmin eder."""
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            return jsonify({"error": "Model veya scaler dosyasi bulunamadi. Once /train calistirin."}), 400

        model = load_model(MODEL_PATH)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)

        rows = fetch_sensor_data()
        if len(rows) < WINDOW_SIZE:
            return jsonify({
                "error": "Tahmin icin en az 10 veri gerekli",
                "current_count": len(rows)
            }), 400

        raw_values = np.array([[float(r["salt"]), float(r["sicaklik"])] for r in rows], dtype=np.float32)
        scaled_values = scaler.transform(raw_values)

        last_window = scaled_values[-WINDOW_SIZE:]
        model_input = np.expand_dims(last_window, axis=0)

        scaled_prediction = model.predict(model_input, verbose=0)
        prediction = scaler.inverse_transform(scaled_prediction)[0]

        return jsonify({
            "predicted_salt": float(prediction[0]),
            "predicted_sicaklik": float(prediction[1]),
            "model_version": MODEL_VERSION,
            "window_size": WINDOW_SIZE,
            "predicted_at": datetime.utcnow().isoformat() + "Z"
        })
    except Exception as exc:
        return jsonify({"error": "Tahmin sirasinda hata", "details": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
