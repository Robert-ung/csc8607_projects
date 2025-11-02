"""
Évaluation — à implémenter.

Doit exposer un main() exécutable via :
    python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best.ckpt

Exigences minimales :
- charger le modèle et le checkpoint
- calculer et afficher/consigner les métriques de test
"""

import argparse
import yaml
from .data_loading import get_dataloaders
from .preprocessing import get_preprocess_transforms
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    # Charger config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Dataloaders (sans augmentation)
    _, _, test_loader, meta = get_dataloaders(config)
    preprocess = get_preprocess_transforms(config)

    # Vérification
    print("Meta:", meta)
    print("Test loader size:", len(test_loader.dataset))

    # Vérification complète
    print("\nDataset sizes:")
    print(f"Test: {len(test_loader.dataset)} examples")
    print(f"Batch size: {test_loader.batch_size}")
    
    # Vérifier un batch
    X_batch, y_batch = next(iter(test_loader))
    print(f"\nShapes: X={X_batch.shape}, y={y_batch.shape}")
    print(f"Value ranges: X=[{X_batch.min():.2f}, {X_batch.max():.2f}]")

if __name__ == "__main__":
    main()