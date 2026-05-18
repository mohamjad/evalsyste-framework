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
        has_neg2 = any(neg in words2 for neg in self.negation_words)
        
        if (has_neg1 or has_neg2) and len(core_words1 & core_words2) > 0:
            overlap_ratio = len(core_words1 & core_words2) / max(len(core_words1), len(core_words2))
            if overlap_ratio > 0.3:  # Significant overlap
                score = max(score, 0.7)
                reasons.append(f"explicit negation with {overlap_ratio:.2f} word overlap")
        
        # Check for direct negation patterns
        for word in core_words1:
            if f"not {word}" in text2 or f"no {word}" in text2:
                score = max(score, 0.9)
                reasons.append(f"direct negation: '{word}'")
        
        reason_str = "; ".join(reasons) if reasons else "no contradiction detected"
        return score, reason_str
    
    def _semantic_contradiction(self, text1: str, text2: str) -> Tuple[float, str]:
        """
        Detect contradictions using semantic embeddings.
        
        Uses cosine similarity between embeddings. Low similarity with
        high semantic relatedness indicates contradiction.
        """
        if not self.model:
            return 0.0, ""
        
        try:
            # Get embeddings
            embeddings = self.model.encode([text1, text2])
            emb1, emb2 = embeddings[0], embeddings[1]
            
            # Cosine similarity
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0, ""
            
            similarity = dot_product / (norm1 * norm2)
            
            # Contradiction score: low similarity suggests contradiction
            # But we need to check if topics are related first
            # For now, use a simple heuristic: similarity < 0.3 suggests contradiction
            contradiction_score = max(0.0, (0.3 - similarity) / 0.3)
            
            reason = f"semantic similarity: {similarity:.3f}"
            return contradiction_score, reason
            
        except Exception as e:
            return 0.0, f"embedding error: {str(e)}"
    
    def calculate_semantic_redundancy(self, texts: List[str]) -> float:
        """
        Calculate semantic redundancy using information theory.
        
        Based on:
        - Shannon entropy for information content
        - Semantic similarity clustering
        
        Returns redundancy ratio (0-1), where 1 = completely redundant
        """
        if len(texts) < 2:
            return 0.0
        
        if self.use_embeddings and self.model:
            return self._embedding_based_redundancy(texts)
        else:
            return self._lexical_redundancy(texts)
    
    def _lexical_redundancy(self, texts: List[str]) -> float:
        """Calculate redundancy using word overlap (Jaccard similarity)."""
        if len(texts) < 2:
            return 0.0
        
        # Tokenize and normalize
        word_sets = []
        for text in texts:
            words = set(re.findall(r'\b\w+\b', text.lower()))
            word_sets.append(words)
        
        # Calculate pairwise Jaccard similarities
        similarities = []
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                intersection = len(word_sets[i] & word_sets[j])
                union = len(word_sets[i] | word_sets[j])
                if union > 0:
                    jaccard = intersection / union
                    similarities.append(jaccard)
        
        if not similarities:
            return 0.0
        
        # Average redundancy
        avg_redundancy = sum(similarities) / len(similarities)
        return avg_redundancy
    
    def _embedding_based_redundancy(self, texts: List[str]) -> float:
        """Calculate redundancy using semantic embeddings."""
        if not self.model or len(texts) < 2:
            return 0.0
        
        try:
            embeddings = self.model.encode(texts)
            
            # Calculate pairwise cosine similarities
            similarities = []
            for i in range(len(embeddings)):
