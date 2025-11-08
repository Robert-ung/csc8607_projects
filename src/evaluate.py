"""
Évaluation — à implémenter.

Doit exposer un main() exécutable via :
    python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best.ckpt

Exigences minimales :
- charger le modèle et le checkpoint
- calculer et afficher/consigner les métriques de test
"""

# Question M9
r"""(.venv) (base) PS C:\Users\rober\Documents\csc8607_projects-main\csc8607_projects-main> python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best.ckpt

Output:

Meta: {'num_classes': 6, 'input_shape': (9, 128)}
Test loader size: 2947

Dataset sizes:
Test: 2947 examples
Batch size: 64

Shapes: X=torch.Size([64, 9, 128]), y=torch.Size([64])
Value ranges: X=[-2.52, 2.03]

=== Résultats sur le jeu de test ===
Loss: 0.1935
Accuracy: 93.72%

Classification Report:
              precision    recall  f1-score   support

           0       1.00      0.95      0.97       496
           1       0.98      0.99      0.99       471
           2       0.94      0.98      0.96       420
           3       0.81      0.91      0.86       491
           4       0.91      0.81      0.86       532
           5       0.99      1.00      1.00       537

    accuracy                           0.94      2947
   macro avg       0.94      0.94      0.94      2947
weighted avg       0.94      0.94      0.94      2947


Confusion Matrix:
[[470   1  25   0   0   0]
 [  0 468   3   0   0   0]
 [  0   7 413   0   0   0]
 [  0   2   0 445  41   3]
 [  0   0   0 103 429   0]
 [  0   0   0   0   0 537]]"""

from .data_loading import get_dataloaders
from .preprocessing import get_preprocess_transforms
import os
import torch
import yaml
import argparse
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from .model import build_model
from .data_loading import get_dataloaders
from .utils import set_seed

def evaluate(config, checkpoint_path, device="cpu"):
    """Évalue le modèle sur le jeu de test."""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Build model and load state
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get test data
    _, _, test_loader, _ = get_dataloaders(config)
    
    # Evaluate
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            
            # Accumulate metrics
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
            # Store predictions for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    
    # Calculate metrics
    test_loss = test_loss / len(test_loader)
    test_acc = 100. * correct / total
    
    # Confusion matrix
    conf_matrix = confusion_matrix(all_targets, all_preds)
    report = classification_report(all_targets, all_preds)
    
    return {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'confusion_matrix': conf_matrix,
        'classification_report': report
    }

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

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Evaluate
    results = evaluate(config, args.checkpoint, device)
    
    # Print results
    print("\n=== Résultats sur le jeu de test ===")
    print(f"Loss: {results['test_loss']:.4f}")
    print(f"Accuracy: {results['test_accuracy']:.2f}%")
    print("\nClassification Report:")
    print(results['classification_report'])
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])

if __name__ == "__main__":
    main()