# Question M2
"""(.venv) (base) PS C:\Users\rober\Documents\csc8607_projects-main\csc8607_projects-main> python -m src.tests.test_first_batch
Output:
=== Premier batch et loss initiale ===
Shape batch X: torch.Size([64, 9, 128])
Shape batch y: torch.Size([64])
Shape logits: torch.Size([64, 6])

Loss initiale: 1.7557
Loss théorique (-log(1/6)): 1.7918
Norme totale des gradients: 17.5777
"""

import torch
import yaml
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model import build_model
from src.data_loading import get_dataloaders
from src.utils import set_seed

def test_first_batch():
    # Load config and set seed
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    set_seed(42)
    
    # Get model and first batch
    model = build_model(config)
    train_loader, _, _, meta = get_dataloaders(config)
    X, y = next(iter(train_loader))
    
    # Forward pass
    logits = model(X)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(logits, y)
    
    # Theoretical loss for uniform predictions
    num_classes = meta["num_classes"]
    theoretical_loss = -np.log(1/num_classes)
    
    # Backward pass
    loss.backward()
    total_grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    
    print("\n=== Premier batch et loss initiale ===")
    print(f"Shape batch X: {X.shape}")
    print(f"Shape batch y: {y.shape}")
    print(f"Shape logits: {logits.shape}")
    print(f"\nLoss initiale: {loss.item():.4f}")
    print(f"Loss théorique (-log(1/6)): {theoretical_loss:.4f}")
    print(f"Norme totale des gradients: {total_grad_norm:.4f}")
    
    return loss.item(), theoretical_loss

if __name__ == "__main__":
    test_first_batch()