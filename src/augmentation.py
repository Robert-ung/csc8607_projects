"""
Data augmentation

Signature imposée :
get_augmentation_transforms(config: dict) -> objet/transform callable (ou None)
"""

import numpy as np

def get_augmentation_transforms(config: dict):
    """Retourne les transformations d'augmentation pour le train uniquement."""

    def augment(x):
        # x : tensor de forme (C, T)

        # Jitter : ajout de bruit gaussien (probabilité 0.5)
        if np.random.rand() < 0.5:
            noise = np.random.normal(0, 0.01, size=x.shape)
            x = x + noise

        # Scaling : variation d’amplitude par canal (probabilité 0.5)
        if np.random.rand() < 0.5:
            scale = np.random.normal(1.0, 0.1, size=(x.shape[0], 1))
            x = x * scale

        return x

    return augment
