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
