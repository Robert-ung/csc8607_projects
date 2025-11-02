import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import torch
from src.data_loading import get_dataloaders
from src.augmentation import get_augmentation_transforms
import yaml

"""
# Charger les labels
y_train = np.loadtxt("data/UCI HAR Dataset/train/y_train.txt") - 1
y_test = np.loadtxt("data/UCI HAR Dataset/test/y_test.txt") - 1

# Split val
from sklearn.model_selection import train_test_split
_, y_val = train_test_split(y_train, test_size=0.2, stratify=y_train, random_state=42)

# Compter les classes
def count_classes(y, name):
    counts = Counter(y)
    print(f"\n{name} class distribution:")
    for cls in sorted(counts):
        print(f"Classe {cls}: {counts[cls]} exemples")

count_classes(y_train, "Train")
count_classes(y_val, "Validation")
count_classes(y_test, "Test")
"""

# Charger config et dataloaders
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

train_loader, _, _, meta = get_dataloaders(config)
augment = get_augmentation_transforms(config)

# Extraire un batch
X_batch, y_batch = next(iter(train_loader))  # (batch_size, C, T)

# Afficher 2 exemples
for i in range(2):
    x = X_batch[i].numpy()  # (C, T)
    y = y_batch[i].item()

    # Appliquer augmentation
    x_aug = augment(x)

    # Afficher un canal (par exemple canal 0)
    plt.figure(figsize=(10, 2))
    plt.plot(x[0], label="Original")
    plt.plot(x_aug[0], label="Augmenté", linestyle="--")
    plt.title(f"Exemple {i} — Label : {y}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"artifacts/exemple_{i}.png")
    plt.close()

print(f"Batch X shape : {X_batch.shape}")  # → torch.Size([64, 9, 128])
print(f"meta['input_shape'] : {meta['input_shape']}")  # → (9, 128)
