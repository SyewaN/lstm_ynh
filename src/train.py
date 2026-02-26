"""TDS LSTM egitim scripti."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from data_loader import load_dataset
from model import build_lstm_model
from preprocessing import preprocess_timeseries


def parse_args() -> argparse.Namespace:
    """CLI argumanlarini parse eder."""
    parser = argparse.ArgumentParser(description="Sulama suyu TDS tahmini icin LSTM egitimi")
    parser.add_argument("--data", type=str, default="data/tds_timeseries.csv", help="Veri dosya yolu")
    parser.add_argument("--sequence-length", type=int, default=24, help="Pencere uzunlugu")
    parser.add_argument("--epochs", type=int, default=50, help="Epoch sayisi")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--output-dir", type=str, default="models", help="Model cikti klasoru")
    return parser.parse_args()


def save_training_plot(history: tf.keras.callbacks.History, output_path: Path) -> None:
    """Egitim ve dogrulama loss grafigini kaydeder."""
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Egitim Gecmisi")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_predictions_plot(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    """Gercek ve tahmin degerlerini ayni grafikte kaydeder."""
    plt.figure(figsize=(12, 5))
    plt.plot(y_true, label="Gercek TDS", linewidth=1.5)
    plt.plot(y_pred, label="Tahmin TDS", linewidth=1.5)
    plt.title("Test Seti: Gercek vs Tahmin")
    plt.xlabel("Zaman adimi")
    plt.ylabel("TDS")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """RMSE, MAE ve R2 skorlarini hesaplar."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return rmse, mae, r2


def save_inference_artifacts(
    output_dir: Path,
    sequence_length: int,
    feature_columns: list[str],
    target_column: str,
    x_scaler: object,
    y_scaler: object,
) -> None:
    """Tahmin asamasinda gerekli metadata ve scaler dosyalarini kaydeder."""
    metadata = {
        "sequence_length": sequence_length,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "specific_conductance_to_tds_ratio": 0.65,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=True, indent=2)

    with open(output_dir / "scalers.pkl", "wb") as f:
        pickle.dump({"x_scaler": x_scaler, "y_scaler": y_scaler}, f)


def main() -> None:
    """Uctan uca egitim akisini calistirir."""
    args = parse_args()

    print("[INFO] TensorFlow surumu:", tf.__version__)
    print("[INFO] Cihazlar:", tf.config.list_physical_devices())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Veri yukleniyor: {args.data}")
    df = load_dataset(args.data)

    feature_columns = ["specific_conductance", "temperature"]
    target_column = "tds"

    print("[INFO] On-isleme basladi (normalizasyon + sliding window + split)")
    (x_train, y_train, x_val, y_val, x_test, y_test), scalers = preprocess_timeseries(
        df=df,
        feature_columns=feature_columns,
        target_column=target_column,
        sequence_length=args.sequence_length,
    )

    print(
        f"[INFO] Split boyutlari -> Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}"
    )

    print("[INFO] LSTM model kuruluyor")
    model = build_lstm_model(input_shape=(x_train.shape[1], x_train.shape[2]))
    model.summary(print_fn=lambda line: print(f"[MODEL] {line}"))

    checkpoint_path = output_dir / "tds_lstm_model.h5"
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print("[INFO] Egitim basliyor")
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    print("[INFO] Test tahminleri uretiliyor")
    y_pred_scaled = model.predict(x_test, verbose=0)

    y_scaler = scalers["y_scaler"]
    y_test_inv = y_scaler.inverse_transform(y_test)
    y_pred_inv = y_scaler.inverse_transform(y_pred_scaled)

    rmse, mae, r2 = compute_metrics(y_test_inv, y_pred_inv)
    metrics_path = output_dir / "metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"R2: {r2:.4f}\n")

    save_training_plot(history=history, output_path=output_dir / "training_history.png")
    save_predictions_plot(
        y_true=y_test_inv.flatten(),
        y_pred=y_pred_inv.flatten(),
        output_path=output_dir / "predictions.png",
    )
    save_inference_artifacts(
        output_dir=output_dir,
        sequence_length=args.sequence_length,
        feature_columns=feature_columns,
        target_column=target_column,
        x_scaler=scalers["x_scaler"],
        y_scaler=scalers["y_scaler"],
    )

    print("[OK] Egitim tamamlandi")
    print(f"[OK] Model: {checkpoint_path}")
    print(f"[OK] Grafikler: {output_dir / 'training_history.png'}, {output_dir / 'predictions.png'}")
    print(f"[OK] Metrikler: {metrics_path}")


if __name__ == "__main__":
    main()
