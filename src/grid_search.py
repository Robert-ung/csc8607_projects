"""
Mini grid search — à implémenter.

Doit exposer un main() exécutable via :
    python -m src.grid_search --config configs/config.yaml

Exigences minimales :
- lire la section 'hparams' de la config
- lancer plusieurs runs en variant les hyperparamètres
- journaliser les hparams et résultats de chaque run (ex: TensorBoard HParams ou équivalent)
"""

# Question M5
r"""(.venv) (base) PS C:\Users\rober\Documents\csc8607_projects-main\csc8607_projects-main> python -m src.grid_search --config configs/config.yaml --max_epochs 5
Output:
Testing configuration: {'lr': 0.00025, 'weight_decay': 1e-05, 'kernel_size': 3, 'num_blocks': [1, 1, 1]}
Val loss: 0.0866, Val accuracy: 96.80%

Testing configuration: {'lr': 0.00025, 'weight_decay': 1e-05, 'kernel_size': 3, 'num_blocks': [2, 2, 2]}
Val loss: 0.0806, Val accuracy: 95.85%

Testing configuration: {'lr': 0.00025, 'weight_decay': 1e-05, 'kernel_size': 5, 'num_blocks': [1, 1, 1]}
Val loss: 0.0926, Val accuracy: 96.67%

Testing configuration: {'lr': 0.00025, 'weight_decay': 1e-05, 'kernel_size': 5, 'num_blocks': [2, 2, 2]}
Val loss: 0.0890, Val accuracy: 96.06%

Testing configuration: {'lr': 0.00025, 'weight_decay': 0.0001, 'kernel_size': 3, 'num_blocks': [1, 1, 1]}
Val loss: 0.0831, Val accuracy: 96.53%

Testing configuration: {'lr': 0.00025, 'weight_decay': 0.0001, 'kernel_size': 3, 'num_blocks': [2, 2, 2]}
Val loss: 0.0940, Val accuracy: 95.79%

Testing configuration: {'lr': 0.00025, 'weight_decay': 0.0001, 'kernel_size': 5, 'num_blocks': [1, 1, 1]}
Val loss: 0.0960, Val accuracy: 96.19%

Testing configuration: {'lr': 0.00025, 'weight_decay': 0.0001, 'kernel_size': 5, 'num_blocks': [2, 2, 2]}
Val loss: 0.0994, Val accuracy: 95.92%

Testing configuration: {'lr': 0.0005, 'weight_decay': 1e-05, 'kernel_size': 3, 'num_blocks': [1, 1, 1]}
Val loss: 0.0922, Val accuracy: 96.60%

Testing configuration: {'lr': 0.0005, 'weight_decay': 1e-05, 'kernel_size': 3, 'num_blocks': [2, 2, 2]}
Val loss: 0.0918, Val accuracy: 95.85%

Testing configuration: {'lr': 0.0005, 'weight_decay': 1e-05, 'kernel_size': 5, 'num_blocks': [1, 1, 1]}
Val loss: 0.1011, Val accuracy: 95.31%

Testing configuration: {'lr': 0.0005, 'weight_decay': 1e-05, 'kernel_size': 5, 'num_blocks': [2, 2, 2]}
Val loss: 0.1099, Val accuracy: 94.09%

Testing configuration: {'lr': 0.0005, 'weight_decay': 0.0001, 'kernel_size': 3, 'num_blocks': [1, 1, 1]}
Val loss: 0.0887, Val accuracy: 96.33%

Testing configuration: {'lr': 0.0005, 'weight_decay': 0.0001, 'kernel_size': 3, 'num_blocks': [2, 2, 2]}
Val loss: 0.0901, Val accuracy: 96.19%

Testing configuration: {'lr': 0.0005, 'weight_decay': 0.0001, 'kernel_size': 5, 'num_blocks': [1, 1, 1]}
Val loss: 0.0978, Val accuracy: 96.06%

Testing configuration: {'lr': 0.0005, 'weight_decay': 0.0001, 'kernel_size': 5, 'num_blocks': [2, 2, 2]}
Val loss: 0.0978, Val accuracy: 95.58%

Testing configuration: {'lr': 0.001, 'weight_decay': 1e-05, 'kernel_size': 3, 'num_blocks': [1, 1, 1]}
Val loss: 0.0948, Val accuracy: 96.46%

Testing configuration: {'lr': 0.001, 'weight_decay': 1e-05, 'kernel_size': 3, 'num_blocks': [2, 2, 2]}
Val loss: 0.0968, Val accuracy: 96.26%

Testing configuration: {'lr': 0.001, 'weight_decay': 1e-05, 'kernel_size': 5, 'num_blocks': [1, 1, 1]}
Val loss: 0.0940, Val accuracy: 95.65%

Testing configuration: {'lr': 0.001, 'weight_decay': 1e-05, 'kernel_size': 5, 'num_blocks': [2, 2, 2]}
Val loss: 0.1029, Val accuracy: 96.33%

Testing configuration: {'lr': 0.001, 'weight_decay': 0.0001, 'kernel_size': 3, 'num_blocks': [1, 1, 1]}
Val loss: 0.0924, Val accuracy: 96.46%

Testing configuration: {'lr': 0.001, 'weight_decay': 0.0001, 'kernel_size': 3, 'num_blocks': [2, 2, 2]}
Val loss: 0.1001, Val accuracy: 96.06%

Testing configuration: {'lr': 0.001, 'weight_decay': 0.0001, 'kernel_size': 5, 'num_blocks': [1, 1, 1]}
Val loss: 0.0974, Val accuracy: 95.24%

Testing configuration: {'lr': 0.001, 'weight_decay': 0.0001, 'kernel_size': 5, 'num_blocks': [2, 2, 2]}
Val loss: 0.0930, Val accuracy: 96.26%

=== Grid Search Results ===

Best configurations by validation loss:

1. Val loss: 0.0806, Val accuracy: 95.85%
Hyperparameters:
  lr: 0.00025
  weight_decay: 1e-05
  kernel_size: 3
  num_blocks: [2, 2, 2]

2. Val loss: 0.0831, Val accuracy: 96.53%
Hyperparameters:
  lr: 0.00025
  weight_decay: 0.0001
  kernel_size: 3
  num_blocks: [1, 1, 1]

3. Val loss: 0.0866, Val accuracy: 96.80%
Hyperparameters:
  lr: 0.00025
  weight_decay: 1e-05
  kernel_size: 3
  num_blocks: [1, 1, 1]"""

import itertools
import torch
from torch.utils.tensorboard import SummaryWriter
import yaml
import argparse
from pathlib import Path
import time

from .model import build_model
from .data_loading import get_dataloaders
from .utils import set_seed, count_parameters

def train_one_config(config, hparams, device="cpu", max_epochs=5):
    """Train one configuration for a few epochs."""
    # Update config with current hparams
    config = config.copy()
    
    # Disable overfit_small for grid search
    if "train" not in config:
        config["train"] = {}
    config["train"]["overfit_small"] = False
    
    # Update optimizer settings
    if "optimizer" not in config["train"]:
        config["train"]["optimizer"] = {}
    config["train"]["optimizer"]["lr"] = float(hparams["lr"])
    config["train"]["optimizer"]["weight_decay"] = float(hparams["weight_decay"])
    
    # Update model settings
    if "model" not in config:
        config["model"] = {}
    config["model"]["kernel_size"] = int(hparams["kernel_size"])
    config["model"]["num_blocks"] = hparams["num_blocks"]
    
    # Model & data
    model = build_model(config).to(device)
    train_loader, val_loader, _, _ = get_dataloaders(config)
    
    # Optimizer & criterion
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hparams["lr"],
        weight_decay=hparams["weight_decay"]
    )
    criterion = torch.nn.CrossEntropyLoss()
    
    # Run name based on hyperparameters
    run_name = (
        f"lr={hparams['lr']:.1e}"
        f"_wd={hparams['weight_decay']:.1e}"
        f"_k{hparams['kernel_size']}"
        f"_b{''.join(map(str, hparams['num_blocks']))}"  # e.g., b111 or b222
    )
    writer = SummaryWriter(f"runs/grid_search/{run_name}")
    
    # Log hyperparameters (convert num_blocks to string for TensorBoard)
    hparams_log = {
        'lr': float(hparams['lr']),
        'weight_decay': float(hparams['weight_decay']),
        'kernel_size': int(hparams['kernel_size']),
        'num_blocks': str(hparams['num_blocks'])  # Convert list to string
    }
    
    writer.add_hparams(
        hparams_log,
        {"hparam/best_val_loss": float('inf')}
    )
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(max_epochs):
        # Train
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
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
        
        # Logging
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/accuracy', val_acc, epoch)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    writer.close()
    return best_val_loss, val_acc

def grid_search(config, max_epochs=5):
    """Perform grid search over hyperparameters defined in config."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config["train"]["seed"])
    
    # Create hyperparameter grid with explicit type conversion
    hparams_grid = {
        'lr': [float(lr) for lr in config["hparams"]["lr"]],
        'weight_decay': [float(wd) for wd in config["hparams"]["weight_decay"]],
        'kernel_size': list(map(int, config["hparams"]["kernel_size"])),
        'num_blocks': config["hparams"]["num_blocks"]
    }
    
    # Generate all combinations
    keys = hparams_grid.keys()
    values = [hparams_grid[key] for key in keys]
    combinations = list(itertools.product(*values))
    
    # Results storage
    results = []
    
    # Run all combinations
    for combo in combinations:
        hparams = dict(zip(keys, combo))
        print(f"\nTesting configuration: {hparams}")
        
        val_loss, val_acc = train_one_config(
            config, hparams, device, max_epochs
        )
        
        results.append({
            'hparams': hparams,
            'val_loss': val_loss,
            'val_accuracy': val_acc
        })
        print(f"Val loss: {val_loss:.4f}, Val accuracy: {val_acc:.2f}%")
    
    # Sort by validation loss
    results.sort(key=lambda x: x['val_loss'])
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max_epochs", type=int, default=5)
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    results = grid_search(config, args.max_epochs)
    
    # Print summary
    print("\n=== Grid Search Results ===")
    print("\nBest configurations by validation loss:")
    for i, result in enumerate(results[:3]):
        print(f"\n{i+1}. Val loss: {result['val_loss']:.4f}, "
              f"Val accuracy: {result['val_accuracy']:.2f}%")
        print("Hyperparameters:")
        for k, v in result['hparams'].items():
            print(f"  {k}: {v}")

if __name__ == "__main__":
    main()