"""
Training Script for Neural Models

Example script showing how to train the contradiction detection,
clarity scoring, and redundancy detection models.

Usage:
    python train_models.py --model contradiction --data data/contradictions.csv
    python train_models.py --model clarity --data data/clarity_ratings.csv
    python train_models.py --model redundancy --data data/redundancy_pairs.csv
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import os

try:
    from neural_models import (
        ContradictionDetectionModel, 
        ClarityScoringModel, 
        RedundancyDetectionModel,
        ModelTrainer
    )
    from semantic_analyzer import SemanticAnalyzer
    NEURAL_AVAILABLE = True
except ImportError:
    print("Error: neural_models not available. Install PyTorch: pip install torch")
    NEURAL_AVAILABLE = False


class ContradictionDataset(Dataset):
    """Dataset for contradiction detection training."""
    
    def __init__(self, pairs: List[Tuple[str, str]], labels: List[int], encoder):
        self.pairs = pairs
        self.labels = labels
        self.encoder = encoder
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        sent1, sent2 = self.pairs[idx]
        label = self.labels[idx]
        
        # Encode sentences
        emb1 = torch.tensor(self.encoder.encode([sent1])[0], dtype=torch.float32)
        emb2 = torch.tensor(self.encoder.encode([sent2])[0], dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return emb1, emb2, label_tensor

