"""Veri yukleme yardimcilari."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["timestamp", "specific_conductance", "temperature", "tds"]


def load_dataset(csv_path: str) -> pd.DataFrame:
    """CSV dosyasini yukler ve temel dogrulama yapar.

    Args:
        csv_path: Veri dosyasi yolu.

    Returns:
        Dogrulanmis ve siralanmis DataFrame.

    Raises:
        FileNotFoundError: Dosya yoksa.
        ValueError: Gerekli kolonlar eksikse veya veri bos ise.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Veri dosyasi bulunamadi: {csv_path}")

    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Eksik kolon(lar): {missing}. Beklenen: {REQUIRED_COLUMNS}")

    if df.empty:
        raise ValueError("Veri dosyasi bos.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Sayisal kolonlardaki eksikler ileri/geri doldurma ile temizlenir.
    num_cols = ["specific_conductance", "temperature", "tds"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    df[num_cols] = df[num_cols].interpolate("linear").bfill().ffill()

    if len(df) < 100:
        raise ValueError(f"Veri satir sayisi cok az: {len(df)}. En az 100 onerilir.")

    return df
