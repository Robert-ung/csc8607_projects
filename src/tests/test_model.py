# Question M1 - Test du modèle
"""(.venv) (base) PS C:\Users\rober\Documents\csc8607_projects-main\csc8607_projects-main> python -m src.tests.test_model
Output : 
Nombre de paramètres: 443,782
Shape d'entrée: torch.Size([32, 9, 128])
Shape de sortie: torch.Size([32, 6])"""

import torch
import yaml
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model import build_model
from src.utils import count_parameters

def test_model():
    # Charger config
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    
    # Construire modèle
    model = build_model(config)
    
    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, 9, 128)  # (B, C, T)
    y = model(x)
    
    # Vérifications
    print(f"Nombre de paramètres: {count_parameters(model):,}")
    print(f"Shape d'entrée: {x.shape}")
    print(f"Shape de sortie: {y.shape}")
    assert y.shape == (batch_size, 6), "Mauvaise shape de sortie"
    
if __name__ == "__main__":
    test_model()