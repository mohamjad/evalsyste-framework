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
