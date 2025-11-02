import unittest
import torch
from src.data_loading import get_dataloaders
import yaml

def test_dataloaders():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    
    # Correction: utiliser train au lieu de train_loader
    train, val, test, meta = get_dataloaders(config)
    
    # 1. Vérifier les shapes
    X, y = next(iter(train))  # Correction ici
    assert X.shape[1:] == tuple(meta["input_shape"])
    assert y.max() < meta["num_classes"]
    
    # 2. Vérifier les tailles des datasets
    print(f"Train size: {len(train.dataset)}")
    print(f"Val size: {len(val.dataset)}")
    print(f"Test size: {len(test.dataset)}")
    
    # 3. Vérifier la normalisation
    print(f"Train mean: {X.mean():.3f}, std: {X.std():.3f}")
    
    # 4. Vérifier les types
    assert X.dtype == torch.float32
    assert y.dtype == torch.long

    # 5. Vérifications supplémentaires
    print("\nVérifications supplémentaires:")
    print(f"Nombre de classes: {meta['num_classes']}")
    print(f"Format d'entrée: {meta['input_shape']}")
    
    # 6. Vérifier batch_size
    assert X.shape[0] == train.batch_size, "Taille de batch incorrecte"

if __name__ == "__main__":
    test_dataloaders()