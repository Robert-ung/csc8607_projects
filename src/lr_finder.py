"""
Recherche de taux d'apprentissage (LR finder) — à implémenter.

Doit exposer un main() exécutable via :
    python -m src.lr_finder --config configs/config.yaml

Exigences minimales :
- produire un log/trace permettant de visualiser (lr, loss) dans TensorBoard ou équivalent.
"""

# Question M4
r"""(.venv) (base) PS C:\Users\rober\Documents\csc8607_projects-main\csc8607_projects-main> python -m src.lr_finder --config configs/config.yaml
Output :
Iter 0: lr = 1.00e-07, loss = 1.7557
Iter 1: lr = 1.10e-07, loss = 1.7339
Iter 2: lr = 1.20e-07, loss = 1.7454
Iter 3: lr = 1.32e-07, loss = 1.7222
Iter 4: lr = 1.45e-07, loss = 1.7770
Iter 5: lr = 1.58e-07, loss = 1.6891
Iter 6: lr = 1.74e-07, loss = 1.7518
Iter 7: lr = 1.91e-07, loss = 1.6944
Iter 8: lr = 2.09e-07, loss = 1.8541
Iter 9: lr = 2.29e-07, loss = 1.6944
Iter 10: lr = 2.51e-07, loss = 1.7642
Iter 11: lr = 2.75e-07, loss = 1.7617
Iter 12: lr = 3.02e-07, loss = 1.7500
Iter 13: lr = 3.31e-07, loss = 1.8221
Iter 14: lr = 3.63e-07, loss = 1.7839
Iter 15: lr = 3.98e-07, loss = 1.7276
(.venv) (base) PS C:\Users\rober\Documents\csc8607_projects-main\csc8607_projects-main> tensorboard --logdir runs
C:\Users\rober\Documents\csc8607_projects-main\csc8607_projects-main\.venv\Lib\site-packages\tensorboard\default.py:30: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  import pkg_resources
TensorFlow installation not found - running with reduced feature set.
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.20.0 at http://localhost:6006/ (Press CTRL+C to quit)
"""

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
import yaml
from pathlib import Path

from .model import build_model
from .data_loading import get_dataloaders
from .utils import set_seed

def find_lr(config, device="cpu", start_lr=1e-7, end_lr=1, num_iter=100):
    """Exécute un LR finder: fait varier le LR exponentiellement et trace la loss."""
    # Désactiver temporairement overfit_small
    config_copy = config.copy()
    if "train" in config_copy:
        config_copy["train"]["overfit_small"] = False
    
    # Setup
    set_seed(config["train"]["seed"])
    model = build_model(config_copy).to(device)
    train_loader, _, _, _ = get_dataloaders(config_copy)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
    
    # Writer
    writer = SummaryWriter("runs/lr_finder")
    
    # Calculate LR multiplication factor
    mult = (end_lr / start_lr) ** (1/num_iter)
    
    # LR finder loop
    lr_list = []
    loss_list = []
    best_loss = float('inf')
    
    model.train()
    iter_count = 0
    while iter_count < num_iter:
        for X, y in train_loader:
            if iter_count >= num_iter:
                break
                
            X, y = X.to(device), y.to(device)
            
            # Forward pass
            outputs = model(X)
            loss = criterion(outputs, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log
            curr_lr = optimizer.param_groups[0]['lr']
            lr_list.append(curr_lr)
            loss_list.append(loss.item())
            
            writer.add_scalar('lr_finder/lr', curr_lr, iter_count)
            writer.add_scalar('lr_finder/loss', loss.item(), iter_count)
            
            print(f"Iter {iter_count}: lr = {curr_lr:.2e}, loss = {loss.item():.4f}")
            
            # Update LR
            for param_group in optimizer.param_groups:
                param_group['lr'] *= mult
                
            # Stop if loss explodes
            if loss.item() > best_loss * 4:
                print("Loss explosion detected, stopping early")
                return lr_list, loss_list
            if loss.item() < best_loss:
                best_loss = loss.item()
                
            iter_count += 1
    
    writer.close()
    return lr_list, loss_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--start_lr", type=float, default=1e-7)
    parser.add_argument("--end_lr", type=float, default=10.0)
    parser.add_argument("--num_iter", type=int, default=200)
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    find_lr(config, device, args.start_lr, args.end_lr, args.num_iter)

if __name__ == "__main__":
    main()