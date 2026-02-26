"""Zaman serisi on-isleme fonksiyonlari."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


SplitArrays = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def create_sliding_windows(features: np.ndarray, target: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Ozellik ve hedef dizilerinden sliding window olusturur.

    Args:
        features: 2D array (n_samples, n_features).
        target: 1D veya 2D array (n_samples,).
        sequence_length: Pencere uzunlugu.

    Returns:
        X ve y dizileri.
    """
    x_windows = []
    y_windows = []

    for idx in range(sequence_length, len(features)):
        x_windows.append(features[idx - sequence_length : idx])
        y_windows.append(target[idx])

    x_array = np.asarray(x_windows, dtype=np.float32)
    y_array = np.asarray(y_windows, dtype=np.float32).reshape(-1, 1)
    return x_array, y_array


def preprocess_timeseries(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    sequence_length: int,
) -> tuple[SplitArrays, Dict[str, MinMaxScaler]]:
    """Veriyi normalize eder, windowing uygular ve 70/15/15 boler.

    Not:
        Data leakage'i azaltmak icin scaler'lar sadece train bolumu uzerinde fit edilir.

    Args:
        df: Ham veri.
        feature_columns: Model girdisi olacak kolonlar.
        target_column: Tahmin edilecek kolon.
        sequence_length: Pencere uzunlugu.

    Returns:
        Split edilmis X/y tuple'i ve scaler sozlugu.
    """
    features = df[feature_columns].values
    target = df[target_column].values.reshape(-1, 1)

    # Once ham veri bazinda split index'leri belirlenir.
    n_total = len(df)
    train_end_raw = int(n_total * 0.70)

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    x_scaler.fit(features[:train_end_raw])
    y_scaler.fit(target[:train_end_raw])

    features_scaled = x_scaler.transform(features)
    target_scaled = y_scaler.transform(target)

    x_all, y_all = create_sliding_windows(features_scaled, target_scaled, sequence_length)

    n_windows = len(x_all)
    train_end = int(n_windows * 0.70)
    val_end = int(n_windows * 0.85)

    x_train, y_train = x_all[:train_end], y_all[:train_end]
    x_val, y_val = x_all[train_end:val_end], y_all[train_end:val_end]
    x_test, y_test = x_all[val_end:], y_all[val_end:]

    splits: SplitArrays = (x_train, y_train, x_val, y_val, x_test, y_test)
    scalers = {"x_scaler": x_scaler, "y_scaler": y_scaler}
    return splits, scalers
