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


class ClarityDataset(Dataset):
    """Dataset for clarity scoring training."""
    
    def __init__(self, statements: List[str], scores: List[float], encoder):
        self.statements = statements
        self.scores = scores
        self.encoder = encoder
    
    def __len__(self):
        return len(self.statements)
    
    def __getitem__(self, idx):
        statement = self.statements[idx]
        score = self.scores[idx]
        
        emb = torch.tensor(self.encoder.encode([statement])[0], dtype=torch.float32)
        score_tensor = torch.tensor(score, dtype=torch.float32)
        
        return emb, score_tensor


def train_contradiction_model(data_path: str, epochs: int = 50):
    """Train contradiction detection model."""
    if not NEURAL_AVAILABLE:
        print("Neural models not available")
        return
    
    # Load data (example - adjust to your format)
    # pairs = [(sent1, sent2), ...]
    # labels = [0 or 1, ...]
    # For now, create dummy data
    print("Note: Replace with your actual data loading")
    pairs = [
        ("system is stable", "system is unstable"),
        ("temperature increased", "temperature decreased"),
        ("operation succeeded", "operation failed"),
        ("system is stable", "system is working"),
        ("temperature is high", "temperature is low"),
    ]
    labels = [1, 1, 1, 0, 1]
    
    # Initialize encoder
    encoder = SemanticAnalyzer(use_embeddings=True)
    if not encoder.model:
        print("Error: sentence-transformers not available")
        return
    
    # Create dataset
    dataset = ContradictionDataset(pairs, labels, encoder.model)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Initialize model
    model = ContradictionDetectionModel(embedding_dim=384, use_pretrained=True)
    trainer = ModelTrainer(model, learning_rate=1e-4)
    
    # Train
    print("Training contradiction detection model...")
    history = trainer.train(train_loader, val_loader, epochs=epochs)
    
    # Save
    os.makedirs("models", exist_ok=True)
    trainer.save_model("models/contradiction_model.pt")
    print(f"Model saved. Best validation loss: {trainer.best_loss:.4f}")
    
    return trainer


def train_clarity_model(data_path: str, epochs: int = 50):
    """Train clarity scoring model."""
    if not NEURAL_AVAILABLE:
        print("Neural models not available")
