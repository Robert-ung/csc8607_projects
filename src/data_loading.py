"""
Chargement des données.

Signature imposée :
get_dataloaders(config: dict) -> (train_loader, val_loader, test_loader, meta: dict)

Le dictionnaire meta doit contenir au minimum :
- "num_classes": int
- "input_shape": tuple (ex: (3, 32, 32) pour des images)
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import os
import torch

class HAR_Dataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # (N, T, C)
        self.X = self.X.permute(0, 2, 1)  # → (N, C, T)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_signals(signal_dir, split):
    signal_types = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
        "total_acc_x", "total_acc_y", "total_acc_z"
    ]
    signals = []
    for signal in signal_types:
        path = os.path.join(signal_dir, f"{signal}_{split}.txt")
        data = np.loadtxt(path)
        signals.append(data[:, :, np.newaxis])  # (N, T, 1)
    return np.concatenate(signals, axis=2)  # (N, T, 9)

def get_dataloaders(config):
    """
    def get_dataloaders(config: dict):
        Crée et retourne les DataLoaders d'entraînement/validation/test et des métadonnées.
        À implémenter.
        raise NotImplementedError("get_dataloaders doit être implémentée par l'étudiant·e.") 
    """
    root = config["dataset"]["root"]
    signal_dir = os.path.join(root, "train", "Inertial Signals")
    test_signal_dir = os.path.join(root, "test", "Inertial Signals")

    # Charger les données brutes
    X_train = load_signals(signal_dir, "train")
    X_test = load_signals(test_signal_dir, "test")

    # Charger les labels
    y_train = np.loadtxt(os.path.join(root, "train", "y_train.txt")) - 1
    y_test = np.loadtxt(os.path.join(root, "test", "y_test.txt")) - 1

    # Normalisation canal par canal
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, 9)
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_train = X_train_flat.reshape(-1, 128, 9)

    X_test_flat = X_test.reshape(-1, 9)
    X_test_flat = scaler.transform(X_test_flat)
    X_test = X_test_flat.reshape(-1, 128, 9)

    # Split validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    # Datasets
    train_dataset = HAR_Dataset(X_train, y_train)
    val_dataset = HAR_Dataset(X_val, y_val)
    test_dataset = HAR_Dataset(X_test, y_test)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # Meta info
    meta = {
        "num_classes": 6,
        "input_shape": (9, 128)
    }

    return train_loader, val_loader, test_loader, meta

