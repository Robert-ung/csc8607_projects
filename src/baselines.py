import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score

# Question M0 : python -m src.baselines --config configs\config.yaml --trials 200 --seed 42
"""Output : Distribution des classes (train): Counter({np.float64(5.0): 1407, np.float64(4.0): 1374, np.float64(3.0): 1286, np.float64(0.0): 1226, np.float64(1.0): 1073, np.float64(2.0): 986})
Classe majoritaire : 5.0
Accuracy : 0.1822
F1-macro : 0.0514

Prédiction aléatoire (100 essais):
Accuracy : 0.1667 ± 0.0069
F1-macro : 0.1664 ± 0.0070"""

# Charger les données
y_train = np.loadtxt("./data/UCI HAR Dataset/train/y_train.txt") - 1  # 0-5
y_test = np.loadtxt("./data/UCI HAR Dataset/test/y_test.txt") - 1    # 0-5

# Distribution des classes (train)
dist = Counter(y_train)
print("Distribution des classes (train):", dist)

# Classe majoritaire (déterminée sur train, évaluée sur test)
major_class = Counter(y_train).most_common(1)[0][0]
y_pred_major = np.full_like(y_test, major_class)
acc_major = accuracy_score(y_test, y_pred_major)
f1_major = f1_score(y_test, y_pred_major, average='macro')
print(f"Classe majoritaire : {major_class}")
print(f"Accuracy : {acc_major:.4f}")
print(f"F1-macro : {f1_major:.4f}")

# Prédiction aléatoire (moyenne sur 100 essais)
num_classes = 6
np.random.seed(42)
acc_random = []
f1_random = []
for _ in range(100):
    y_pred_random = np.random.randint(0, num_classes, size=len(y_test))
    acc_random.append(accuracy_score(y_test, y_pred_random))
    f1_random.append(f1_score(y_test, y_pred_random, average='macro'))

print(f"\nPrédiction aléatoire (100 essais):")
print(f"Accuracy : {np.mean(acc_random):.4f} ± {np.std(acc_random):.4f}")
print(f"F1-macro : {np.mean(f1_random):.4f} ± {np.std(f1_random):.4f}")