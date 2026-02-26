"""LSTM model mimarisi."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.optimizers import Adam


def build_lstm_model(input_shape: tuple[int, int]) -> tf.keras.Model:
    """Istenen mimariye uygun LSTM modelini kurar.

    Mimari:
        LSTM(50, return_sequences=True) + Dropout(0.2)
        LSTM(50) + Dropout(0.2)
        Dense(1)

    Args:
        input_shape: (sequence_length, n_features)

    Returns:
        Derlenmis Keras model.
    """
    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1),
        ]
    )

    model.compile(optimizer=Adam(), loss="mse")
    return model
