"""USGS API'den veri ceken, basarisiz olursa sentetik veri ureten script."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import requests

USGS_IV_URL = "https://waterservices.usgs.gov/nwis/iv/"


def fetch_usgs_data(
    site_id: str,
    start_dt: str,
    end_dt: str,
    timeout: int = 20,
) -> pd.DataFrame:
    """USGS anlik veri servisi uzerinden specific conductance degerlerini ceker.

    Args:
        site_id: USGS istasyon kimligi.
        start_dt: Baslangic tarihi (YYYY-MM-DD).
        end_dt: Bitis tarihi (YYYY-MM-DD).
        timeout: HTTP timeout suresi (sn).

    Returns:
        timestamp ve specific_conductance kolonlarini iceren DataFrame.

    Raises:
        RuntimeError: API yaniti beklenen formatta degilse.
        requests.RequestException: HTTP seviyesinde hata olursa.
    """
    params = {
        "format": "json",
        "sites": site_id,
        "parameterCd": "00095",  # Specific Conductance
        "startDT": start_dt,
        "endDT": end_dt,
        "siteStatus": "all",
    }
    print(f"[INFO] USGS API cagrisi basladi. site_id={site_id}, start={start_dt}, end={end_dt}")
    response = requests.get(USGS_IV_URL, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    series = payload.get("value", {}).get("timeSeries", [])
    if not series:
        raise RuntimeError("USGS API timeSeries bos dondu.")

    points = series[0].get("values", [{}])[0].get("value", [])
    if not points:
        raise RuntimeError("USGS API olcum noktasi donmedi.")

    rows = []
    for point in points:
        ts_raw = point.get("dateTime")
        val_raw = point.get("value")
        if ts_raw is None or val_raw in (None, ""):
            continue
        try:
            rows.append((pd.to_datetime(ts_raw), float(val_raw)))
        except ValueError:
            continue

    if not rows:
        raise RuntimeError("USGS verisi parse edilemedi veya tamamen eksik.")

    df = pd.DataFrame(rows, columns=["timestamp", "specific_conductance"])
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    print(f"[INFO] USGS'den {len(df)} satir veri alindi.")
    return df


def generate_synthetic_data(num_points: int = 8760, seed: int = 42) -> pd.DataFrame:
    """TDS icin sentetik saatlik zaman serisi olusturur.

    Formul:
        TDS(t) = 750 + 50*sin(2*pi*t/8760) + 10*sin(2*pi*t/24) + noise

    Args:
        num_points: Uretilecek saatlik veri adedi.
        seed: Tekrarlanabilirlik icin random seed.

    Returns:
        timestamp, tds ve 4 ozellik kolonunu iceren DataFrame.
    """
    print("[WARN] USGS kullanilamadi, sentetik veri uretiliyor.")
    rng = np.random.default_rng(seed)
    t = np.arange(num_points)

    tds = 750 + 50 * np.sin(2 * np.pi * t / 8760) + 10 * np.sin(2 * np.pi * t / 24)
    tds = tds + rng.normal(0, 5, size=num_points)

    # Specific conductance -> TDS yaklasik donusum: TDS ~= 0.65 * SC
    specific_conductance = tds / 0.65

    # Sicaklikta gunluk ve mevsimsel dalga + az miktar gurultu
    temperature = 18 + 8 * np.sin(2 * np.pi * t / 24) + 6 * np.sin(2 * np.pi * t / 8760)
    temperature = temperature + rng.normal(0, 0.8, size=num_points)
    weather_index = 55 + 25 * np.sin(2 * np.pi * t / 168) + rng.normal(0, 2.0, size=num_points)
    soil_type_code = np.full(num_points, 2.0, dtype=float)

    start = pd.Timestamp("2025-01-01 00:00:00")
    timestamps = pd.date_range(start=start, periods=num_points, freq="h")

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "specific_conductance": specific_conductance.astype(float),
            "temperature": temperature.astype(float),
            "weather_index": weather_index.astype(float),
            "soil_type_code": soil_type_code.astype(float),
            "tds": tds.astype(float),
            "data_source": "synthetic",
        }
    )
    print(f"[INFO] Sentetik veri olusturuldu. satir={len(df)}")
    return df


def build_dataset_from_usgs(
    site_id: str,
    start_dt: str,
    end_dt: str,
    synthetic_points: int,
) -> pd.DataFrame:
    """USGS veri setini olusturur, hata olursa sentetik veriye duser."""
    try:
        usgs_df = fetch_usgs_data(site_id=site_id, start_dt=start_dt, end_dt=end_dt)

        # USGS verisini saatlige indirger ve eksikleri lineer doldurur.
        usgs_df = usgs_df.set_index("timestamp").resample("h").mean().interpolate("time").reset_index()
        usgs_df["tds"] = usgs_df["specific_conductance"] * 0.65

        # Gercek sicaklik API'den alinmadigi icin sentetik ama gercekci bir profil eklenir.
        t = np.arange(len(usgs_df))
        temp = 18 + 8 * np.sin(2 * np.pi * t / 24) + np.random.default_rng(123).normal(0, 1.0, len(usgs_df))
        weather_idx = 55 + 25 * np.sin(2 * np.pi * t / 168) + np.random.default_rng(456).normal(0, 2.0, len(usgs_df))
        usgs_df["temperature"] = temp
        usgs_df["weather_index"] = weather_idx
        usgs_df["soil_type_code"] = 2.0
        usgs_df["data_source"] = "usgs"

        print(f"[INFO] USGS tabanli veri seti hazir. satir={len(usgs_df)}")
        return usgs_df[
            [
                "timestamp",
                "specific_conductance",
                "temperature",
                "weather_index",
                "soil_type_code",
                "tds",
                "data_source",
            ]
        ]

    except (requests.RequestException, RuntimeError, ValueError) as exc:
        print(f"[WARN] USGS veri cekimi basarisiz: {exc}")
        return generate_synthetic_data(num_points=synthetic_points)


def parse_args() -> argparse.Namespace:
    """CLI argumanlarini parse eder."""
    parser = argparse.ArgumentParser(description="USGS veya sentetik TDS veri uretici")
    parser.add_argument("--site-id", type=str, default="09380000", help="USGS site id")
    parser.add_argument("--start-date", type=str, default="2025-01-01", help="Baslangic tarihi")
    parser.add_argument("--end-date", type=str, default="2025-12-31", help="Bitis tarihi")
    parser.add_argument("--synthetic-points", type=int, default=8760, help="Fallback sentetik nokta sayisi")
    parser.add_argument("--output", type=str, default="data/tds_timeseries.csv", help="Cikti CSV yolu")
    return parser.parse_args()


def main() -> None:
    """Script giris noktasi."""
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_dataset_from_usgs(
        site_id=args.site_id,
        start_dt=args.start_date,
        end_dt=args.end_date,
        synthetic_points=args.synthetic_points,
    )

    df.to_csv(output_path, index=False)
    print(f"[OK] Veri kaydedildi: {output_path}")


if __name__ == "__main__":
    main()
