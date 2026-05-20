import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def load_fundium_data(path):
    df = pd.read_csv(path)
    df["new_time"] = pd.to_datetime(df["new_time"])
    return df


def load_price_data(path):
    df = pd.read_csv(path, encoding="utf-8-sig", skiprows=1)
    df.columns = ["timestamp", "renewable_gen_mw", "load_mw", "price_eur_mwh"]
    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_convert(None)
    df["price_eur_mwh"] = pd.to_numeric(df["price_eur_mwh"], errors="coerce")
    df["renewable_gen_mw"] = pd.to_numeric(df["renewable_gen_mw"], errors="coerce")
    df["load_mw"] = pd.to_numeric(df["load_mw"], errors="coerce")
    return df


def create_time_features(df):
    dt = df["new_time"] if "new_time" in df.columns else df["timestamp"]
    df["month"] = dt.dt.month
    df["day_of_week"] = dt.dt.dayofweek
    df["hour"] = dt.dt.hour
    df["minute"] = dt.dt.minute

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    return df


def get_time_features(dt_series, price_series=None):
    data = {
        "month_sin": np.sin(2 * np.pi * dt_series.dt.month / 12),
        "month_cos": np.cos(2 * np.pi * dt_series.dt.month / 12),
        "dow_sin": np.sin(2 * np.pi * dt_series.dt.dayofweek / 7),
        "dow_cos": np.cos(2 * np.pi * dt_series.dt.dayofweek / 7),
        "hour_sin": np.sin(2 * np.pi * dt_series.dt.hour / 24),
        "hour_cos": np.cos(2 * np.pi * dt_series.dt.hour / 24),
    }
    return pd.DataFrame(data)


def merge_datasets(fundium_path, price_path):
    fundium = load_fundium_data(fundium_path)
    fundium = create_time_features(fundium)

    price = load_price_data(price_path)
    price["timestamp"] = pd.to_datetime(price["timestamp"])

    price = price.dropna(subset=["price_eur_mwh"])
    price = price.sort_values("timestamp").reset_index(drop=True)

    merged = pd.merge_asof(
        fundium.sort_values("new_time"),
        price.sort_values("timestamp"),
        left_on="new_time",
        right_on="timestamp",
        direction="nearest",
    )

    dla_cols = [
        "dla_stromverbrauch_kwh",
        "pl2_stromverbrauch_kwh",
        "dla_volume_out_m3_per_m",
        "dla_master_ae1_pressure_bar",
        "dla_volume_out_m3",
        "ofen_abwaerme_nestle_5893_mw",
    ]
    for col in dla_cols:
        if col in merged.columns:
            merged[col] = merged[col].ffill().bfill()

    merged["price_eur_mwh"] = merged["price_eur_mwh"].ffill().bfill()
    merged["renewable_gen_mw"] = merged["renewable_gen_mw"].ffill().bfill()
    merged["load_mw"] = merged["load_mw"].ffill().bfill()

    return merged.dropna(subset=["dla_stromverbrauch_kwh", "pl2_stromverbrauch_kwh", "price_eur_mwh", "ofen_abwaerme_nestle_5893_mw"])


class DLADataset(Dataset):
    def __init__(self, df, time_cols, dla_cols, target_col):
        self.time_features = torch.tensor(df[time_cols].values, dtype=torch.float32)
        self.dla_features = torch.tensor(df[dla_cols].values, dtype=torch.float32)
        self.target = torch.tensor(df[target_col].values, dtype=torch.float32).log()

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return (self.time_features[idx], self.dla_features[idx], self.target[idx])


class PriceDataset(Dataset):
    def __init__(self, df, time_cols, target_col):
        self.time_features = torch.tensor(df[time_cols].values, dtype=torch.float32)
        self.target = torch.tensor(df[target_col].values, dtype=torch.float32).log()

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return (self.time_features[idx], self.target[idx])
