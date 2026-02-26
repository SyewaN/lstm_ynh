"""Egitilmis LSTM modeli ile ileri adim TDS tahmini."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from data_loader import load_dataset


def parse_args() -> argparse.Namespace:
    """CLI argumanlarini parse eder."""
    parser = argparse.ArgumentParser(description="Egitilmis modelle TDS tahmini")
    parser.add_argument("--data", type=str, default="data/tds_timeseries.csv", help="Girdi veri yolu")
    parser.add_argument("--model-path", type=str, default="models/tds_lstm_model.h5", help="Egitilmis model yolu")
    parser.add_argument("--metadata-path", type=str, default="models/metadata.json", help="Metadata dosyasi")
    parser.add_argument("--scalers-path", type=str, default="models/scalers.pkl", help="Scaler dosyasi")
    parser.add_argument("--steps", type=int, default=1, help="Kac adim ileri tahmin yapilacagi")
    parser.add_argument("--output", type=str, default="models/forecast.csv", help="Tahmin cikti CSV yolu")
    return parser.parse_args()


def load_artifacts(metadata_path: Path, scalers_path: Path) -> tuple[dict, object, object]:
    """Metadata ve scaler artefactlarini yukler."""
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata bulunamadi: {metadata_path}")
    if not scalers_path.exists():
        raise FileNotFoundError(f"Scaler dosyasi bulunamadi: {scalers_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    with open(scalers_path, "rb") as f:
        scalers = pickle.load(f)

    return metadata, scalers["x_scaler"], scalers["y_scaler"]


def forecast_next_steps(
    model: tf.keras.Model,
    df: pd.DataFrame,
    feature_columns: list[str],
    sequence_length: int,
    x_scaler: object,
    y_scaler: object,
    steps: int,
    sc_to_tds_ratio: float,
) -> pd.DataFrame:
    """Son pencereden baslayarak ileri adim TDS tahmini yapar."""
    if len(df) < sequence_length:
        raise ValueError(f"Tahmin icin en az {sequence_length} satir gerekiyor, mevcut: {len(df)}")
    if steps < 1:
        raise ValueError("steps en az 1 olmali")

    history_features = df[feature_columns].astype(float).values.copy()
    last_timestamp = pd.to_datetime(df["timestamp"].iloc[-1])
    last_feature_values = {col: float(df[col].iloc[-1]) for col in feature_columns}

    rows: list[dict] = []

    for _ in range(steps):
        window = history_features[-sequence_length:]
        x_scaled = x_scaler.transform(window)
        x_input = np.expand_dims(x_scaled, axis=0).astype(np.float32)

        y_pred_scaled = model.predict(x_input, verbose=0)
        tds_pred = float(y_scaler.inverse_transform(y_pred_scaled)[0, 0])
        sc_pred = tds_pred / sc_to_tds_ratio

        last_timestamp = last_timestamp + pd.Timedelta(hours=1)
        row = {"timestamp": last_timestamp, "predicted_tds": tds_pred}

        next_feature_map = dict(last_feature_values)
        if "specific_conductance" in next_feature_map:
            next_feature_map["specific_conductance"] = sc_pred

        for col, val in next_feature_map.items():
            row[f"assumed_{col}"] = val
        rows.append(row)

        next_features = np.array([next_feature_map[col] for col in feature_columns], dtype=np.float32)
        history_features = np.vstack([history_features, next_features])
        last_feature_values = next_feature_map

    return pd.DataFrame(rows)


def main() -> None:
    """Tahmin scripti giris noktasi."""
    args = parse_args()

    model_path = Path(args.model_path)
    metadata_path = Path(args.metadata_path)
    scalers_path = Path(args.scalers_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Veri yukleniyor: {args.data}")
    df = load_dataset(args.data)

    print(f"[INFO] Model yukleniyor: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    metadata, x_scaler, y_scaler = load_artifacts(metadata_path, scalers_path)

    sequence_length = int(metadata["sequence_length"])
    feature_columns = list(metadata["feature_columns"])
    ratio = float(metadata.get("specific_conductance_to_tds_ratio", 0.65))

    print(f"[INFO] Tahmin basladi. steps={args.steps}")
    forecast_df = forecast_next_steps(
        model=model,
        df=df,
        feature_columns=feature_columns,
        sequence_length=sequence_length,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        steps=args.steps,
        sc_to_tds_ratio=ratio,
    )
    forecast_df.to_csv(output_path, index=False)

    print(f"[OK] Tahmin kaydedildi: {output_path}")
    print(f"[OK] Sonraki saat TDS tahmini: {forecast_df['predicted_tds'].iloc[0]:.3f}")


if __name__ == "__main__":
    main()
