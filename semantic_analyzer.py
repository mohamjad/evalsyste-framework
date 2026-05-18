"""
Semantic Analysis Module for Advanced Contradiction Detection

Provides optional embedding-assisted and lexical heuristics for
contradiction and redundancy checks.
"""

import math
from typing import List, Tuple, Optional
from collections import Counter
import re

# Optional: use sentence transformers if available
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    SentenceTransformer = None
    np = None


class SemanticAnalyzer:
    """
    Optional semantic-analysis helper.

    Uses multiple techniques:
    1. Embedding-based semantic similarity (cosine similarity)
    2. Lexical contradiction patterns (antonyms, negations)
    3. Information-theoretic measures
    """
    
    def __init__(self, use_embeddings: bool = True):
        """
        Initialize semantic analyzer.
        
        Args:
            use_embeddings: If True and sentence-transformers available, use embeddings
        """
        self.use_embeddings = use_embeddings and SEMANTIC_AVAILABLE
        self.model = None
        
        if self.use_embeddings:
            try:
                # Use lightweight model for production
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                print(f"Warning: Could not load sentence transformer: {e}")
                self.use_embeddings = False
        
        # Hand-authored antonym patterns used as heuristic fallbacks.
        self.antonym_patterns = [
            ("increase", "decrease"), ("increase", "reduce"), ("increase", "decline"),
            ("stable", "unstable"), ("stable", "volatile"), ("stable", "erratic"),
            ("succeed", "fail"), ("success", "failure"), ("working", "broken"),
            ("correct", "incorrect"), ("correct", "wrong"), ("true", "false"),
            ("positive", "negative"), ("good", "bad"), ("high", "low"),
            ("fast", "slow"), ("large", "small"), ("more", "less"),
            ("enable", "disable"), ("allow", "prevent"), ("accept", "reject"),
        ]
        
        # Negation markers
        self.negation_words = ["not", "no", "never", "none", "neither", "nobody", "nothing"]
    
    def detect_contradiction(self, text1: str, text2: str) -> Tuple[bool, float, str]:
        """
        Detect if two texts contradict each other.
        
        Returns:
            (is_contradiction, confidence, reason)
            - is_contradiction: bool
            - confidence: float 0-1
            - reason: explanation string
        """
        text1_lower = text1.lower()
        text2_lower = text2.lower()
