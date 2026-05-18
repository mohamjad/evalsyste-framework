"""
Neural Network Models for Learned Contradiction Detection and Clarity Scoring

Optional PyTorch modules for experiments that go beyond the default
heuristic paths in the main framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import math

# Optional: use transformers if available
try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None
    AutoModel = None


class ContradictionDetectionModel(nn.Module):
    """
    Neural network for contradiction detection.
    
    Architecture:
    - Sentence encoder (BERT-based or MLP)
    - Interaction layer (concatenation + MLP)
    - Classification head (binary: contradiction or not)
    
    Can be trained end-to-end on contradiction datasets.
    """
    
    def __init__(self, 
                 embedding_dim: int = 384,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 use_pretrained: bool = True):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.use_pretrained = use_pretrained and TRANSFORMERS_AVAILABLE
        
        # Sentence encoder
        if self.use_pretrained:
            # Use pre-trained BERT/Sentence-BERT as encoder
            self.encoder = None  # Will be initialized separately
            self.encoder_dim = 384  # all-MiniLM-L6-v2 dimension
        else:
            # Learn embeddings from scratch
            self.encoder = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.encoder_dim = embedding_dim
        
        # Interaction layer: combine two sentence embeddings
        interaction_input_dim = self.encoder_dim * 2 + 1  # +1 for cosine similarity
        layers = []
        current_dim = interaction_input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        
        self.interaction = nn.Sequential(*layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization for learned layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def encode_sentence(self, sentence_embedding: torch.Tensor) -> torch.Tensor:
        """Encode a sentence to fixed-size representation."""
        if self.use_pretrained:
            # Assume sentence_embedding is already from pre-trained model
            return sentence_embedding
        else:
            return self.encoder(sentence_embedding)
    
    def forward(self, 
                sent1_emb: torch.Tensor, 
                sent2_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for contradiction detection.
        
        Args:
            sent1_emb: Sentence 1 embedding [batch_size, embedding_dim]
            sent2_emb: Sentence 2 embedding [batch_size, embedding_dim]
        
        Returns:
            Contradiction probability [batch_size, 1]
        """
        # Encode sentences
        enc1 = self.encode_sentence(sent1_emb)
        enc2 = self.encode_sentence(sent2_emb)
        
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(enc1, enc2, dim=1, keepdim=True)
        
        # Concatenate features
        combined = torch.cat([enc1, enc2, cos_sim], dim=1)
        
        # Interaction layer
        interaction_out = self.interaction(combined)
        
        # Classification
        contradiction_prob = self.classifier(interaction_out)
        
        return contradiction_prob
    
    def predict(self, sent1_emb: torch.Tensor, sent2_emb: torch.Tensor, 
                threshold: float = 0.5) -> Tuple[bool, float]:
        """Predict contradiction with confidence."""
        self.eval()
        with torch.no_grad():
            prob = self.forward(sent1_emb, sent2_emb)
            is_contradiction = (prob.item() > threshold)
            confidence = prob.item()
        return is_contradiction, confidence


class ClarityScoringModel(nn.Module):
    """
    Neural network for learned clarity scoring.
    
    Takes statement text/embeddings and predicts clarity score (0-1).
    Can be trained on human-annotated clarity ratings.
    """
    
    def __init__(self,
                 input_dim: int = 384,
                 hidden_dims: List[int] = [256, 128],
                 dropout: float = 0.1):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        
        # Output: clarity score (0-1)
        layers.append(nn.Linear(current_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, statement_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for clarity scoring.
        
        Args:
            statement_embedding: Statement embedding [batch_size, input_dim]
        
        Returns:
            Clarity score [batch_size, 1]
        """
        return self.network(statement_embedding)
    
    def predict(self, statement_embedding: torch.Tensor) -> float:
        """Predict clarity score."""
        self.eval()
        with torch.no_grad():
            score = self.forward(statement_embedding)
            return score.item()


class RedundancyDetectionModel(nn.Module):
    """
    Neural network for learned redundancy detection.
    
    Takes multiple statement embeddings and predicts redundancy ratio.
    """
    
    def __init__(self,
                 embedding_dim: int = 384,
                 hidden_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        
        # Pairwise similarity network
        self.pairwise_net = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 1, hidden_dim),  # +1 for cosine sim
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, statement_embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for redundancy calculation.
        
        Args:
            statement_embeddings: List of statement embeddings
        
        Returns:
            Redundancy score (0-1)
        """
        if len(statement_embeddings) < 2:
            return torch.tensor(0.0)
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(statement_embeddings)):
            for j in range(i + 1, len(statement_embeddings)):
                emb1 = statement_embeddings[i]
                emb2 = statement_embeddings[j]
                
                cos_sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
                combined = torch.cat([emb1, emb2, cos_sim.unsqueeze(0)])
                
                sim_score = self.pairwise_net(combined.unsqueeze(0))
                similarities.append(sim_score)
        
        if not similarities:
            return torch.tensor(0.0)
        
        # Average pairwise similarity = redundancy
        redundancy = torch.stack(similarities).mean()
        return redundancy
    
    def predict(self, statement_embeddings: List[torch.Tensor]) -> float:
        """Predict redundancy score."""
        self.eval()
        with torch.no_grad():
            score = self.forward(statement_embeddings)
            return score.item()


class ModelTrainer:
    """
    Training utilities for neural models.
    
    Implements:
    - Training loops with proper batching
    - Loss functions (BCE for classification, MSE for regression)
    - Optimizers (Adam with learning rate scheduling)
    - Early stopping
    - Model checkpointing
    """
    
    def __init__(self, 
                 model: nn.Module,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 device: Optional[torch.device] = None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        self.best_loss = float('inf')
        self.training_history = []
    
    def train_epoch(self, dataloader, criterion):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(*batch[:-1])  # All but last element is target
            targets = batch[-1].to(self.device)
            
            # Calculate loss
            loss = criterion(outputs.squeeze(), targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def validate(self, dataloader, criterion):
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                outputs = self.model(*batch[:-1])
                targets = batch[-1].to(self.device)
                loss = criterion(outputs.squeeze(), targets)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def train(self, train_loader, val_loader, epochs: int = 50, 
              early_stopping_patience: int = 10):
        """Full training loop with early stopping."""
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, nn.BCELoss())
            val_loss = self.validate(val_loader, nn.BCELoss())
            
            self.scheduler.step(val_loss)
            
            self.training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss
            })
            
            # Early stopping
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    break
        
        return self.training_history
    
    def save_model(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'training_history': self.training_history
        }, path)
    
    def load_model(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_loss = checkpoint['best_loss']
        self.training_history = checkpoint['training_history']
