"""
Pré-traitements.

Signature imposée :
get_preprocess_transforms(config: dict) -> objet/transform callable
"""
import numpy as np
import torch

def get_preprocess_transforms(config: dict):
    """Retourne les transformations de pré-traitement."""
    def preprocess(x):
        # Conversion en tensor si nécessaire
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        # Normalisation par canal si spécifiée dans config
        if config.get("preprocess", {}).get("normalize"):
            mean = config["preprocess"]["normalize"].get("mean", [0.0])
            std = config["preprocess"]["normalize"].get("std", [1.0])
            x = (x - torch.tensor(mean)[:, None]) / torch.tensor(std)[:, None]
            
        return x
    
    return preprocess