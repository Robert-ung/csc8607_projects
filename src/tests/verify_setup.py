import os
import yaml
import torch
import numpy as np
import sys
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

from src.data_loading import get_dataloaders
from src.preprocessing import get_preprocess_transforms
from src.augmentation import get_augmentation_transforms

def verify_setup():
    # 1. Vérifier les chemins
    print("=== Vérification des chemins ===")
    paths = {
        "config": PROJECT_ROOT / "configs" / "config.yaml",
        "runs": PROJECT_ROOT / "runs",
        "artifacts": PROJECT_ROOT / "artifacts"
    }
    for name, path in paths.items():
        assert path.exists(), f"{name} manquant: {path}"
        print(f"✓ {name} trouvé: {path}")
    
    # 2. Charger et vérifier la configuration
    with open(paths["config"]) as f:
        config = yaml.safe_load(f)
    print("\n=== Configuration chargée ===")
    assert config["dataset"]["name"] == "UCI_HAR", "Dataset incorrect"
    dataset_path = PROJECT_ROOT / config["dataset"]["root"].lstrip("./")
    assert dataset_path.exists(), f"Dataset manquant: {dataset_path}"
    print(f"✓ Dataset trouvé: {dataset_path}")
    
    # 3. Vérifier les dataloaders
    print("\n=== Vérification des dataloaders ===")
    train, val, test, meta = get_dataloaders(config)
    
    # Vérifier les tailles
    sizes = {
        "Train": len(train.dataset),
        "Val": len(val.dataset),
        "Test": len(test.dataset)
    }
    for name, size in sizes.items():
        print(f"✓ {name}: {size} exemples")
    
    # 4. Vérifier les données et transformations
    print("\n=== Vérification des données et transformations ===")
    X, y = next(iter(train))
    
    # Vérifier les shapes
    expected_shape = tuple(meta["input_shape"])
    assert X.shape[1:] == expected_shape, f"Shape incorrecte: {X.shape[1:]} vs {expected_shape}"
    print(f"✓ Shape correcte: {X.shape}")
    
    # Vérifier les types
    assert X.dtype == torch.float32, f"Type X incorrect: {X.dtype}"
    assert y.dtype == torch.int64, f"Type y incorrect: {y.dtype}"
    print("✓ Types corrects")
    
    # Vérifier les labels
    assert y.min() >= 0 and y.max() < meta["num_classes"], "Labels hors limites"
    print(f"✓ Labels dans [0, {meta['num_classes']-1}]")
    
    # Vérifier la normalisation
    print(f"✓ Statistiques train: mean={X.mean():.3f}, std={X.std():.3f}")
    
    # 5. Vérifier les transformations
    preprocess = get_preprocess_transforms(config)
    augment = get_augmentation_transforms(config)
    
    if augment:
        X_aug = augment(X[0])
        assert X_aug.shape == X[0].shape, "L'augmentation modifie la shape"
        print("✓ Augmentation préserve la shape")
    
    print("\n✅ Toutes les vérifications sont passées!")

if __name__ == "__main__":
    # Create required directories
    for dir_path in ["runs", "artifacts"]:
        os.makedirs(PROJECT_ROOT / dir_path, exist_ok=True)
    verify_setup()