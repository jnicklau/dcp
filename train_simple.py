import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from data_loader import merge_datasets, get_time_features
from config import *


class GaussianBNN(nn.Module):
    """BNN predicting mean and variance in original space"""

    def __init__(self, input_dim, hidden_dims=[64, 32], init_noise=0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)

        with torch.no_grad():
            self.network[-1].bias.data[1] = torch.tensor([init_noise])

    def forward(self, x):
        out = self.network(x)
        log_var = out[:, 1]
        log_var = log_var.clamp(-5, 5)
        sigma = torch.exp(0.5 * log_var) + 0.01
        return out[:, 0], sigma

    def nll_loss(self, x, y):
        mu, sigma = self.forward(x)
        return torch.mean(0.5 * torch.log(sigma**2) + (y - mu) ** 2 / (2 * sigma**2))

    def predict(self, x, n_samples=NUM_MC_SAMPLES):
        self.train()
        mus = []
        sigmas = []
        with torch.no_grad():
            for _ in range(n_samples):
                mu, sigma = self.forward(x)
                mus.append(mu)
                sigmas.append(sigma)

        mu_samples = torch.stack(mus).cpu().numpy()
        sigma_samples = torch.stack(sigmas).cpu().numpy()

        mu_mean = mu_samples.mean(axis=0)
        mu_std = mu_samples.std(axis=0)
        sigma_mean = sigma_samples.mean(axis=0)
        sigma_std = sigma_samples.std(axis=0)

        self.eval()
        return mu_mean, sigma_mean, mu_std, sigma_std


class NormalizedBNN(nn.Module):
    """BNN for normalized targets [0,1] (e.g., price)"""

    def __init__(self, input_dim, hidden_dims=[64, 32], init_noise=0.1):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)

        with torch.no_grad():
            self.network[-1].bias.data[1] = torch.tensor([init_noise])

    def forward(self, x):
        out = self.network(x)
        mu = torch.sigmoid(out[:, 0])
        log_var = out[:, 1].clamp(-5, 5)
        sigma = torch.exp(0.5 * log_var) + 0.01
        return mu, sigma

    def nll_loss(self, x, y):
        mu, sigma = self.forward(x)
        return torch.mean(0.5 * torch.log(sigma**2) + (y - mu) ** 2 / (2 * sigma**2))

    def predict(self, x, n_samples=NUM_MC_SAMPLES):
        self.train()
        mus = []
        sigmas = []
        with torch.no_grad():
            for _ in range(n_samples):
                mu, sigma = self.forward(x)
                mus.append(mu)
                sigmas.append(sigma)

        mu_samples = torch.stack(mus).cpu().numpy()
        sigma_samples = torch.stack(sigmas).cpu().numpy()

        mu_mean = mu_samples.mean(axis=0)
        mu_std = mu_samples.std(axis=0)
        sigma_mean = sigma_samples.mean(axis=0)
        sigma_std = sigma_samples.std(axis=0)

        self.eval()
        return mu_mean, sigma_mean, mu_std, sigma_std

    def predict_denormalized(self, x, n_samples=NUM_MC_SAMPLES, price_min=0, price_max=1):
        mu_norm, sigma_norm, mu_std, sigma_std = self.predict(x, n_samples)
        price_range = price_max - price_min
        mu_actual = mu_norm * price_range + price_min
        sigma_actual = sigma_norm * price_range
        mu_std_actual = mu_std * price_range
        sigma_std_actual = sigma_std * price_range
        return mu_actual, sigma_actual, mu_std_actual, sigma_std_actual


def train_model_nll(
    model, train_loader, epochs=NUM_EPOCHS, lr=LEARNING_RATE, name="Model"
):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    pbar = tqdm(range(epochs), desc=f"Training {name}")
    losses = []
    for epoch in pbar:
        epoch_loss = 0
        n_batches = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = model.nll_loss(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    return model, losses


def calibrate_sigma(errors, std_dev, target_coverage=0.95):
    """Find scale factor to achieve target coverage using quantile matching"""
    abs_errors = np.abs(errors)
    target_quantile = target_coverage
    actual_quantile = np.percentile(abs_errors, target_quantile * 100)
    std_mean = np.mean(std_dev)

    if std_mean > 0:
        scale_factor = actual_quantile / (1.96 * std_mean)
    else:
        scale_factor = 1.0

    return np.clip(scale_factor, 1.0, 5.0)


def calculate_metrics(y_actual, y_pred, mu_std, sigma_mean, confidence=0.95):
    z_score = 1.96 if confidence == 0.95 else 1.645

    rmse = np.sqrt(np.mean((y_pred - y_actual) ** 2))
    mae = np.mean(np.abs(y_pred - y_actual))
    mape = np.mean(np.abs((y_pred - y_actual) / (y_actual + 1e-8))) * 100

    total_std = np.sqrt(mu_std**2 + sigma_mean**2)

    scale = calibrate_sigma(y_actual - y_pred, total_std, target_coverage=confidence)
    total_std_cal = total_std * scale

    lower = y_pred - z_score * total_std_cal
    upper = y_pred + z_score * total_std_cal
    coverage = np.mean((y_actual >= lower) & (y_actual <= upper)) * 100

    crps = np.mean(np.abs(y_pred - y_actual)) + np.mean(total_std_cal) / 3

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "coverage": coverage,
        "crps": crps,
        "total_std_mean": total_std_cal.mean(),
        "scale_factor": scale,
    }


def prepare_data(merged_df):
    time_feats = get_time_features(merged_df["new_time"])
    dla_target = merged_df["dla_stromverbrauch_kwh"].values
    price_target = merged_df["price_eur_mwh"].values
    abwaerme_target = merged_df["ofen_abwaerme_nestle_5893_mw"].values

    renewable = merged_df["renewable_gen_mw"].values
    load = merged_df["load_mw"].values
    price_cols = np.column_stack([renewable, load])

    price_cols = (price_cols - price_cols.mean(axis=0)) / (
        price_cols.std(axis=0) + 1e-8
    )

    train_size = int(len(merged_df) * TRAIN_SPLIT)

    return (
        {
            "time": time_feats.values[:train_size],
            "price_cols": price_cols[:train_size],
            "dla_target": dla_target[:train_size],
            "price_target": price_target[:train_size],
            "abwaerme_target": abwaerme_target[:train_size],
        },
        {
            "time": time_feats.values[train_size:],
            "price_cols": price_cols[train_size:],
            "dla_target": dla_target[train_size:],
            "price_target": price_target[train_size:],
            "abwaerme_target": abwaerme_target[train_size:],
        },
        train_size,
    )


def main():
    print("=" * 60)
    print("DLA Stochastic Optimization - Training")
    print("=" * 60)

    print("\n[1/5] Loading data...")
    merged = merge_datasets(FUNDIUM_DATA_PATH, PRICE_DATA_PATH)
    print(f"  Merged dataset: {merged.shape[0]} samples")

    print("\n[2/5] Preparing train/test split...")
    train_data, test_data, train_size = prepare_data(merged)
    print(
        f"  Train: {len(train_data['time'])} samples, Test: {len(test_data['time'])} samples"
    )

    dla_mean = train_data["dla_target"].mean()
    dla_std = train_data["dla_target"].std()
    dla_train_norm = (train_data["dla_target"] - dla_mean) / dla_std

    print(f"  DLA: mean={dla_mean:.1f} kWh, std={dla_std:.1f} kWh")
    print(f"  price: mean={train_data['price_target'].mean():.1f} €/MWh, std={train_data['price_target'].std():.1f} €/MWh")
    print(f"  abwaerme: mean={train_data['abwaerme_target'].mean():.1f} MW, std={train_data['abwaerme_target'].std():.1f} MW")

    print("\n[3/5] Training DLA BNN (Gaussian NLL in original space)...")
    dla_dataset = TensorDataset(
        torch.tensor(train_data["time"], dtype=torch.float32),
        torch.tensor(dla_train_norm, dtype=torch.float32),
    )
    dla_loader = DataLoader(dla_dataset, batch_size=256, shuffle=True)

    dla_bnn = GaussianBNN(6, HIDDEN_DIMS, init_noise=0.5)
    dla_bnn, dla_losses = train_model_nll(dla_bnn, dla_loader, name="DLA BNN")
    torch.save(dla_bnn.state_dict(), "dla_bnn.pt")
    np.savez("dla_norm.npz", mean=dla_mean, std=dla_std)

    print("\n[4/5] Training Price BNN (with renewable + load features)...")
    price_min = train_data["price_target"].min()
    price_max = train_data["price_target"].max()
    price_targets = (train_data["price_target"] - price_min) / (price_max - price_min)

    price_features = np.hstack([train_data["time"], train_data["price_cols"]])
    price_input_dim = 8

    price_dataset = TensorDataset(
        torch.tensor(price_features, dtype=torch.float32),
        torch.tensor(price_targets, dtype=torch.float32),
    )
    price_loader = DataLoader(price_dataset, batch_size=256, shuffle=True)

    price_bnn = NormalizedBNN(price_input_dim, HIDDEN_DIMS, init_noise=0.1)
    price_bnn, price_losses = train_model_nll(price_bnn, price_loader, name="Price BNN")
    torch.save(price_bnn.state_dict(), "price_bnn.pt")
    np.savez("price_norm.npz", price_min=price_min, price_max=price_max)
    print(f"  Price range: [{price_min:.2f}, {price_max:.2f}] EUR/MWh")

    print("\n[5/6] Training Abwaerme Nestle BNN...")
    abwaerme_mean = train_data["abwaerme_target"].mean()
    abwaerme_std = train_data["abwaerme_target"].std()
    abwaerme_train_norm = (train_data["abwaerme_target"] - abwaerme_mean) / abwaerme_std

    abwaerme_dataset = TensorDataset(
        torch.tensor(train_data["time"], dtype=torch.float32),
        torch.tensor(abwaerme_train_norm, dtype=torch.float32),
    )
    abwaerme_loader = DataLoader(abwaerme_dataset, batch_size=256, shuffle=True)

    abwaerme_bnn = GaussianBNN(6, HIDDEN_DIMS, init_noise=0.5)
    abwaerme_bnn, abwaerme_losses = train_model_nll(abwaerme_bnn, abwaerme_loader, name="Abwaerme BNN")
    torch.save(abwaerme_bnn.state_dict(), "abwaerme_bnn.pt")
    np.savez("abwaerme_norm.npz", mean=abwaerme_mean, std=abwaerme_std)
    print(f"  Abwaerme: mean={abwaerme_mean:.3f} MW, std={abwaerme_std:.3f} MW")

    print("\n[6/6] Plotting training losses...")
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(dla_losses) + 1)
    ax.plot(epochs, dla_losses, "b-", label="DLA BNN", linewidth=2)
    ax.plot(epochs, price_losses, "r-", label="Price BNN", linewidth=2)
    ax.plot(epochs, abwaerme_losses, "g-", label="Abwaerme BNN", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NLL Loss")
    ax.set_title("Training Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_loss.png", dpi=150)
    print(f"  Saved to: training_loss.png")

    print("\n[5/5] Test Set Evaluation...")
    print("-" * 60)

    test_dla_x = torch.tensor(test_data["time"], dtype=torch.float32)
    test_dla_y = test_data["dla_target"]

    dla_mu_norm, dla_sigma_norm, dla_mu_std, _ = dla_bnn.predict(
        test_dla_x, n_samples=NUM_MC_SAMPLES
    )
    dla_mu = dla_mu_norm * dla_std + dla_mean
    dla_sigma = dla_sigma_norm * dla_std
    dla_mu_std = dla_mu_std * dla_std

    dla_metrics = calculate_metrics(test_dla_y, dla_mu, dla_mu_std, dla_sigma)
    print("\nDLA Consumption Test Metrics:")
    print(f"  RMSE:      {dla_metrics['rmse']:.2f} kWh")
    print(f"  MAE:       {dla_metrics['mae']:.2f} kWh")
    print(f"  MAPE:      {dla_metrics['mape']:.1f}%")
    print(f"  Coverage:  {dla_metrics['coverage']:.1f}% (95% CI)")
    print(f"  Total Std: {dla_metrics['total_std_mean']:.2f} kWh")

    test_price_features = np.hstack([test_data["time"], test_data["price_cols"]])
    test_price_x = torch.tensor(test_price_features, dtype=torch.float32)
    test_price_y = test_data["price_target"]

    price_mu, price_sigma, price_mu_std, _ = price_bnn.predict_denormalized(
        test_price_x, n_samples=NUM_MC_SAMPLES, price_min=price_min, price_max=price_max
    )

    price_metrics = calculate_metrics(test_price_y, price_mu, price_mu_std, price_sigma)
    print("\nPrice Day-Ahead Test Metrics:")
    print(f"  RMSE:      {price_metrics['rmse']:.2f} EUR/MWh")
    print(f"  MAE:       {price_metrics['mae']:.2f} EUR/MWh")
    print(f"  MAPE:      {price_metrics['mape']:.1f}%")
    print(f"  Coverage:  {price_metrics['coverage']:.1f}% (95% CI)")
    print(f"  Total Std: {price_metrics['total_std_mean']:.2f} EUR/MWh")

    test_abwaerme_x = torch.tensor(test_data["time"], dtype=torch.float32)
    test_abwaerme_y = test_data["abwaerme_target"]

    abwaerme_mu_norm, abwaerme_sigma_norm, abwaerme_mu_std, _ = abwaerme_bnn.predict(
        test_abwaerme_x, n_samples=NUM_MC_SAMPLES
    )
    abwaerme_mu = abwaerme_mu_norm * abwaerme_std + abwaerme_mean
    abwaerme_sigma = abwaerme_sigma_norm * abwaerme_std
    abwaerme_mu_std = abwaerme_mu_std * abwaerme_std

    abwaerme_metrics = calculate_metrics(test_abwaerme_y, abwaerme_mu, abwaerme_mu_std, abwaerme_sigma)
    print("\nAbwaerme Nestle Test Metrics:")
    print(f"  RMSE:      {abwaerme_metrics['rmse']:.4f} MW")
    print(f"  MAE:       {abwaerme_metrics['mae']:.4f} MW")
    print(f"  MAPE:      {abwaerme_metrics['mape']:.1f}%")
    print(f"  Coverage:  {abwaerme_metrics['coverage']:.1f}% (95% CI)")
    print(f"  Total Std: {abwaerme_metrics['total_std_mean']:.4f} MW")

    print("\n" + "=" * 60)
    print("Training complete! Models saved.")
    print("=" * 60)


if __name__ == "__main__":
    main()
