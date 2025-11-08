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

# Question M3 
r"""(.venv) (base) PS C:\Users\rober\Documents\csc8607_projects-main\csc8607_projects-main> python -m src.train --config configs/config.yaml --overfit_small
Output :
=== Mode overfit_small activé ===
⚠️ Mode overfit_small : train réduit à 32 exemples
Epoch 1: train/loss = 1.7402
Epoch 2: train/loss = 0.6875
Epoch 3: train/loss = 0.3316
Epoch 4: train/loss = 0.1891
Epoch 5: train/loss = 0.1190
Epoch 6: train/loss = 0.0767
Epoch 7: train/loss = 0.0469
Epoch 8: train/loss = 0.0270
Epoch 9: train/loss = 0.0166
Epoch 10: train/loss = 0.0115
(.venv) (base) PS C:\Users\rober\Documents\csc8607_projects-main\csc8607_projects-main> tensorboard --logdir runs
C:\Users\rober\Documents\csc8607_projects-main\csc8607_projects-main\.venv\Lib\site-packages\tensorboard\default.py:30: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  import pkg_resources
TensorFlow installation not found - running with reduced feature set.
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.20.0 at http://localhost:6006/ (Press CTRL+C to quit)"""

import argparse
import yaml
import torch
from torch.utils.tensorboard import SummaryWriter
from .data_loading import get_dataloaders
from .preprocessing import get_preprocess_transforms
from .augmentation import get_augmentation_transforms
import os

import argparse
import yaml
import torch
import os
from torch.utils.tensorboard import SummaryWriter

from .model import build_model
from .utils import count_parameters, set_seed

def train_overfit(config, device="cpu"):
    """
    Entraînement sur un très petit sous-ensemble (mode overfit_small).
    Permet de vérifier que le modèle peut sur-apprendre.
    """
    # Initialisation
    set_seed(config["train"]["seed"])
    model = build_model(config).to(device)
    train_loader, _, _, _ = get_dataloaders(config)

    # Optimiseur et loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["train"]["optimizer"]["lr"],
        weight_decay=config["train"]["optimizer"]["weight_decay"]
    )
    criterion = torch.nn.CrossEntropyLoss()

    # TensorBoard writer
    writer = SummaryWriter(config["paths"]["runs_dir"])

    # Boucle d'entraînement
    for epoch in range(config["train"]["epochs"]):
        model.train()
        epoch_loss = 0.0

        for i, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Log à chaque batch
            writer.add_scalar('train/loss', loss.item(), epoch * len(train_loader) + i)

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}: train/loss = {avg_loss:.4f}")

    writer.close()
    return model

def main():
    """
    Point d'entrée principal du script d'entraînement.
    Gère les arguments, la config, les chemins, et lance l'entraînement.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overfit_small", action="store_true")
    args = parser.parse_args()

    # Charger la config YAML
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Appliquer les arguments dans la config
    config["train"]["seed"] = args.seed
    config["train"]["overfit_small"] = args.overfit_small

    # Créer les dossiers runs/ et artifacts/
    os.makedirs(config["paths"]["runs_dir"], exist_ok=True)
    os.makedirs(config["paths"]["artifacts_dir"], exist_ok=True)

    # Détection du device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Mode overfit_small
    if args.overfit_small:
        print("=== Mode overfit_small activé ===")
        train_overfit(config, device)
    else:
        raise NotImplementedError("Mode entraînement complet non encore implémenté.")

if __name__ == "__main__":
    main()