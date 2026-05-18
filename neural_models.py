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

