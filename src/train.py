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

# Question M6
r"""(.venv) (base) PS C:\Users\rober\Documents\csc8607_projects-main\csc8607_projects-main> python -m src.train --config configs/config.yaml
Output : 
Epoch 0: train_loss = 0.2607, val_loss = 0.1382, val_acc = 94.29%
Epoch 1: train_loss = 0.1290, val_loss = 0.0924, val_acc = 96.33%
Epoch 2: train_loss = 0.1263, val_loss = 0.0979, val_acc = 94.29%
Epoch 3: train_loss = 0.1167, val_loss = 0.0806, val_acc = 96.19%
Epoch 4: train_loss = 0.1102, val_loss = 0.0864, val_acc = 95.99%
Epoch 5: train_loss = 0.1062, val_loss = 0.0842, val_acc = 96.19%
Epoch 6: train_loss = 0.1061, val_loss = 0.1203, val_acc = 94.22%
Epoch 7: train_loss = 0.1097, val_loss = 0.0857, val_acc = 96.33%
Epoch 8: train_loss = 0.0982, val_loss = 0.0884, val_acc = 95.79%
Epoch 9: train_loss = 0.0968, val_loss = 0.0724, val_acc = 96.94%
Epoch 10: train_loss = 0.0896, val_loss = 0.0662, val_acc = 96.74%
Epoch 11: train_loss = 0.0953, val_loss = 0.0675, val_acc = 97.14%
Epoch 12: train_loss = 0.0939, val_loss = 0.0784, val_acc = 96.40%
Epoch 13: train_loss = 0.0935, val_loss = 0.0714, val_acc = 96.94%
Epoch 14: train_loss = 0.0931, val_loss = 0.0749, val_acc = 95.92%
(.venv) (base) PS C:\Users\rober\Documents\csc8607_projects-main\csc8607_projects-main> tensorboard --logdir runs
C:\Users\rober\Documents\csc8607_projects-main\csc8607_projects-main\.venv\Lib\site-packages\tensorboard\default.py:30: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  import pkg_resources
TensorFlow installation not found - running with reduced feature set.
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.20.0 at http://localhost:6006/ (Press CTRL+C to quit)"""

import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
import yaml
import os
from pathlib import Path

from .model import build_model
from .data_loading import get_dataloaders
from .utils import set_seed

def train_full(config, device="cpu"):
    """Complete training with best configuration."""
    # Setup
    set_seed(config["train"]["seed"])
    model = build_model(config).to(device)
    train_loader, val_loader, _, _ = get_dataloaders(config)
    
    # Optimizer & criterion
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["train"]["optimizer"]["lr"],
        weight_decay=config["train"]["optimizer"]["weight_decay"]
    )
    criterion = torch.nn.CrossEntropyLoss()
    
    # TensorBoard writer
    run_name = (
        f"full_train_lr={config['train']['optimizer']['lr']:.1e}"
        f"_wd={config['train']['optimizer']['weight_decay']:.1e}"
        f"_k{config['model']['kernel_size']}"
        f"_b{''.join(map(str, config['model']['num_blocks']))}"
    )
    writer = SummaryWriter(f"runs/{run_name}")
    
    # Training loop
    best_val_loss = float('inf')
    best_state_dict = None
    
    for epoch in range(config["train"]["epochs"]):
        # Train
        model.train()
        train_loss = 0
        for i, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * correct / total
        
        # Logging
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/accuracy', val_acc, epoch)
        
        print(f"Epoch {epoch}: train_loss = {train_loss:.4f}, "
              f"val_loss = {val_loss:.4f}, val_acc = {val_acc:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = model.state_dict()
    
    # Save best checkpoint
    if best_state_dict is not None:
        checkpoint_path = Path(config["paths"]["artifacts_dir"]) / "best.ckpt"
        torch.save({
            'model_state_dict': best_state_dict,
            'config': config,
            'val_loss': best_val_loss
        }, checkpoint_path)
    
    writer.close()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Best configuration from grid search
    config["train"]["optimizer"]["lr"] = 2.5e-4
    config["train"]["optimizer"]["weight_decay"] = 1e-5
    config["model"]["kernel_size"] = 3
    config["model"]["num_blocks"] = [2, 2, 2]
    config["train"]["epochs"] = 15  # 10-20 epochs
    
    # Create directories
    os.makedirs(config["paths"]["runs_dir"], exist_ok=True)
    os.makedirs(config["paths"]["artifacts_dir"], exist_ok=True)
    
    # Train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_full(config, device)

if __name__ == "__main__":
    main()