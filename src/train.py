"""
Entraînement principal (à implémenter par l'étudiant·e).

Doit exposer un main() exécutable via :
    python -m src.train --config configs/config.yaml [--seed 42]

Exigences minimales :
- lire la config YAML
- respecter les chemins 'runs/' et 'artifacts/' définis dans la config
- journaliser les scalars 'train/loss' et 'val/loss' (et au moins une métrique de classification si applicable)
- supporter le flag --overfit_small (si True, sur-apprendre sur un très petit échantillon)
"""

import argparse
import yaml
import torch
from torch.utils.tensorboard import SummaryWriter
from .data_loading import get_dataloaders
from .preprocessing import get_preprocess_transforms
from .augmentation import get_augmentation_transforms
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Charger config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Seed
    torch.manual_seed(args.seed)
    
    # Dataloaders et transforms
    train_loader, val_loader, test_loader, meta = get_dataloaders(config)
    preprocess = get_preprocess_transforms(config)
    augment = get_augmentation_transforms(config)

    # Vérification rapide
    print("Meta:", meta)
    batch = next(iter(train_loader))
    print("Batch shapes:", batch[0].shape, batch[1].shape)

    # Créer les dossiers nécessaires
    os.makedirs(config["paths"]["runs_dir"], exist_ok=True)
    os.makedirs(config["paths"]["artifacts_dir"], exist_ok=True)
    
    # Initialiser writer TensorBoard
    writer = SummaryWriter(config["paths"]["runs_dir"])
    
    # Vérification des shapes et types
    X_batch, y_batch = next(iter(train_loader))
    print(f"Batch shapes: X={X_batch.shape}, y={y_batch.shape}")
    print(f"Types: X={X_batch.dtype}, y={y_batch.dtype}")
    print(f"Labels range: {y_batch.min().item()}-{y_batch.max().item()}")

if __name__ == "__main__":
    main()

