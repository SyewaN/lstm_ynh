import os
from pathlib import Path

import joblib
import mysql.connector
import numpy as np
import pandas as pd
from flask import Flask, jsonify
from flask_cors import CORS
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model

app = Flask(__name__)
CORS(app)

DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_USER = os.environ.get("DB_USER", "esp32user")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
DB_NAME = os.environ.get("DB_NAME", "esp32monitor")

MODEL_DIR = Path("/var/www/esp32monitor/lstm/models/")
SEQUENCE_LENGTH = 10
MODEL_PATH = MODEL_DIR / "model.h5"
SCALER_PATH = MODEL_DIR / "scaler.pkl"


def get_connection():
    # Her islem icin yeni baglanti acilarak servis kararliligi korunur.
    return mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
    )


def create_sequences(values, seq_length):
    x_data, y_data = [], []
    for i in range(seq_length, len(values)):
        x_data.append(values[i - seq_length : i])
        y_data.append(values[i])
    return np.array(x_data), np.array(y_data)


@app.route("/health", methods=["GET"])
def health():
    try:
        return jsonify({"status": "ok"})
    except Exception as exc:
        return jsonify({"error": f"Saglik kontrolu hatasi: {str(exc)}"}), 500


@app.route("/train", methods=["GET"])
def train():
    conn = None
    cursor = None

    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT salt, sicaklik FROM sensor_data ORDER BY timestamp ASC, id ASC"
        )
        rows = cursor.fetchall()

        data_count = len(rows)
        if data_count < 50:
            return jsonify({"error": "Yetersiz veri", "count": data_count}), 400

        df = pd.DataFrame(rows, columns=["salt", "sicaklik"])
        values = df[["salt", "sicaklik"]].astype("float32").values

        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(values)

        x_train, y_train = create_sequences(scaled_values, SEQUENCE_LENGTH)
        if len(x_train) == 0:
            return jsonify({"error": "Egitim dizisi olusturulamadi", "count": data_count}), 400

        model = Sequential(
            [
                LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, 2)),
                Dropout(0.2),
                LSTM(32),
                Dropout(0.2),
                Dense(2),
            ]
        )
        model.compile(optimizer="adam", loss="mse")

        history = model.fit(x_train, y_train, epochs=50, batch_size=16, verbose=0)

        model.save(MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)

        final_loss = float(history.history["loss"][-1])
        return jsonify(
            {
                "status": "ok",
                "loss": final_loss,
                "data_count": data_count,
            }
        )
    except Exception as exc:
        return jsonify({"error": f"Model egitimi sirasinda hata: {str(exc)}"}), 500
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None and conn.is_connected():
            conn.close()


@app.route("/predict", methods=["GET"])
def predict():
    conn = None
    cursor = None

    try:
        if not MODEL_PATH.exists() or not SCALER_PATH.exists():
            return jsonify({"error": "Model henüz eğitilmedi"}), 400

        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT salt, sicaklik FROM sensor_data ORDER BY timestamp DESC, id DESC LIMIT %s",
            (SEQUENCE_LENGTH,),
        )
        rows = cursor.fetchall()

        if len(rows) < SEQUENCE_LENGTH:
            return jsonify({"error": "Tahmin için yeterli veri yok", "count": len(rows)}), 400

        rows.reverse()
        df = pd.DataFrame(rows, columns=["salt", "sicaklik"])
        values = df[["salt", "sicaklik"]].astype("float32").values

        scaler = joblib.load(SCALER_PATH)
        model = load_model(MODEL_PATH)

        scaled_values = scaler.transform(values)
        x_input = np.expand_dims(scaled_values, axis=0)

        pred_scaled = model.predict(x_input, verbose=0)
        pred = scaler.inverse_transform(pred_scaled)

        predicted_salt = float(pred[0][0])
        predicted_sicaklik = float(pred[0][1])

        cursor.execute(
            "INSERT INTO predictions (predicted_salt, predicted_sicaklik) VALUES (%s, %s)",
            (predicted_salt, predicted_sicaklik),
        )
        conn.commit()

        return jsonify(
            {
                "predicted_salt": round(predicted_salt, 2),
                "predicted_sicaklik": round(predicted_sicaklik, 2),
            }
        )
    except Exception as exc:
        return jsonify({"error": f"Tahmin sirasinda hata: {str(exc)}"}), 500
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None and conn.is_connected():
            conn.close()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
