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
