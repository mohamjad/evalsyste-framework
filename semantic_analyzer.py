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
        
        # Method 1: Lexical pattern matching (fast, reliable)
        lexical_score, lexical_reason = self._lexical_contradiction(text1_lower, text2_lower)
        
        # Method 2: Semantic similarity (if embeddings available)
        semantic_score = 0.0
        semantic_reason = ""
        if self.use_embeddings:
            semantic_score, semantic_reason = self._semantic_contradiction(text1, text2)
        
        # Combine scores with weights
        if self.use_embeddings:
            # Weighted combination: 60% semantic, 40% lexical
            combined_score = 0.6 * semantic_score + 0.4 * lexical_score
        else:
            combined_score = lexical_score
        
        is_contradiction = combined_score > 0.6  # Threshold based on empirical testing
        
        reason = f"{lexical_reason}"
        if semantic_reason:
            reason += f" | {semantic_reason}"
        
        return is_contradiction, combined_score, reason
    
    def _lexical_contradiction(self, text1: str, text2: str) -> Tuple[float, str]:
        """
        Detect contradictions using lexical patterns.
        
        Based on a small hand-authored list of antonym and negation cues.
        """
        score = 0.0
        reasons = []
        
        # Check for antonym pairs
        for ant1, ant2 in self.antonym_patterns:
            has_ant1_in_1 = ant1 in text1
            has_ant2_in_2 = ant2 in text2
            has_ant2_in_1 = ant2 in text1
            has_ant1_in_2 = ant1 in text2
            
            if (has_ant1_in_1 and has_ant2_in_2) or (has_ant2_in_1 and has_ant1_in_2):
                score = max(score, 0.8)
                reasons.append(f"antonym pair: '{ant1}' vs '{ant2}'")
        
        # Check for explicit negation
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        # Remove negation markers and compare
        core_words1 = words1 - set(self.negation_words)
        core_words2 = words2 - set(self.negation_words)
        
        # If one has negation and core words overlap significantly
        has_neg1 = any(neg in words1 for neg in self.negation_words)
