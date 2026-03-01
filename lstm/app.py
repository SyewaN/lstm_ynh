import os
from datetime import datetime

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
# Tüm originler için CORS açık.
CORS(app)

DB_HOST = os.environ.get('DB_HOST', 'localhost')
DB_USER = os.environ.get('DB_USER', 'esp32user')
DB_PASSWORD = os.environ.get('DB_PASSWORD', '')
DB_NAME = os.environ.get('DB_NAME', 'esp32monitor')

MODEL_DIR = '/var/www/esp32monitor/lstm/models/'
MODEL_PATH = MODEL_DIR + 'model.h5'
SCALER_PATH = MODEL_DIR + 'scaler.pkl'
SEQUENCE_LENGTH = 10

# Model klasörü yoksa oluştur.
os.makedirs(MODEL_DIR, exist_ok=True)


def get_db_connection():
    return mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
    )


def build_lstm_model():
    # İki özellikli (salt, sicaklik) zaman serisi için LSTM mimarisi.
    model = Sequential(
        [
            LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, 2)),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(2),
        ]
    )
    model.compile(optimizer='adam', loss='mse')
    return model


def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i - sequence_length : i])
        y.append(data[i])
    return np.array(X), np.array(y)


@app.get('/health')
def health():
    try:
        return jsonify({'status': 'ok'})
    except Exception as exc:
        return jsonify({'error': 'Sağlık kontrolü başarısız', 'detail': str(exc)}), 500


@app.get('/train')
def train():
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute('SELECT salt, sicaklik FROM sensor_data ORDER BY timestamp ASC')
        rows = cursor.fetchall()

        data_count = len(rows)
        if data_count < 50:
            return jsonify({'error': 'Yetersiz veri', 'count': data_count}), 400

        df = pd.DataFrame(rows)
        values = df[['salt', 'sicaklik']].astype(float).values

        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(values)

        X, y = create_sequences(scaled_values, SEQUENCE_LENGTH)
        if len(X) == 0:
            return jsonify({'error': 'Sequence oluşturulamadı', 'count': data_count}), 400

        model = build_lstm_model()
        history = model.fit(X, y, epochs=50, batch_size=16, verbose=0)

        model.save(MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)

        final_loss = float(history.history['loss'][-1])
        return jsonify(
            {
                'status': 'ok',
                'loss': final_loss,
                'data_count': data_count,
                'epochs': 50,
            }
        )
    except Exception as exc:
        return jsonify({'error': 'Model eğitimi başarısız', 'detail': str(exc)}), 500
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@app.get('/predict')
def predict():
    conn = None
    cursor = None
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            return jsonify({'error': 'Model henüz eğitilmedi'}), 400

        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            'SELECT salt, sicaklik FROM sensor_data ORDER BY timestamp DESC LIMIT 10'
        )
        rows = cursor.fetchall()

        if len(rows) < 10:
            return jsonify({'error': 'Tahmin için en az 10 kayıt gerekli', 'count': len(rows)}), 400

        # DESC gelen veriyi kronolojik sıraya çevir.
        rows.reverse()

        values = np.array([[float(r['salt']), float(r['sicaklik'])] for r in rows])
        scaled_values = scaler.transform(values)
        X_input = scaled_values.reshape(1, SEQUENCE_LENGTH, 2)

        predicted_scaled = model.predict(X_input, verbose=0)
        predicted_values = scaler.inverse_transform(predicted_scaled)[0]

        predicted_salt = float(predicted_values[0])
        predicted_sicaklik = float(predicted_values[1])
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        cursor.execute(
            'INSERT INTO predictions (predicted_salt, predicted_sicaklik) VALUES (%s, %s)',
            (predicted_salt, predicted_sicaklik),
        )
        conn.commit()

        return jsonify(
            {
                'predicted_salt': round(predicted_salt, 2),
                'predicted_sicaklik': round(predicted_sicaklik, 2),
                'timestamp': now_str,
            }
        )
    except Exception as exc:
        return jsonify({'error': 'Tahmin başarısız', 'detail': str(exc)}), 500
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=False)
