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

