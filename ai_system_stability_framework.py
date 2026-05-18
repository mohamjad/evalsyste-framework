"""
AI System Stability & Signal Amplification Framework

A lightweight monitoring toolkit for tracking contradiction, clarity,
and stability signals as context scales.

Version: 2.0
Date: February 7, 2026
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict
import math

# Optional advanced modules
try:
    from semantic_analyzer import SemanticAnalyzer
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    SemanticAnalyzer = None

try:
    from statistical_analysis import StatisticalAnalyzer
    STATISTICAL_AVAILABLE = True
except ImportError:
    STATISTICAL_AVAILABLE = False
    StatisticalAnalyzer = None

try:
    import torch
    from neural_models import ContradictionDetectionModel, ClarityScoringModel, RedundancyDetectionModel
    NEURAL_AVAILABLE = True
except ImportError:
    NEURAL_AVAILABLE = False
    ContradictionDetectionModel = None
    ClarityScoringModel = None
    RedundancyDetectionModel = None
    torch = None


@dataclass
class Statement:
    """A single claim or assertion made by the system."""
    content: str
    timestamp: float
    context_size: int
    confidence: float = 1.0
    dependencies: List[str] = field(default_factory=list)
    operation_id: Optional[str] = None


@dataclass
class Contradiction:
    """A detected logical conflict between statements."""
    statement1: Statement
    statement2: Statement
    reason: str
    detected_at: float


@dataclass
class ContextSnapshot:
    """Snapshot of system state at a particular context size."""
    context_size: int
    coherence: float
    operations_count: int
    error_rate: float = 0.0
    timestamp: float = field(default_factory=time.time)
    statements_count: int = 0


class AuditLogger:
    """
    Structured logging for framework calculations and warnings.
    """
    
    def __init__(self, log_file: str = "stability.log", log_level: str = "INFO"):
        self.log_file = log_file
        self.logger = logging.getLogger("ai_stability")
        self.logger.setLevel(getattr(logging, log_level))
        
        # File handler
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s]\n'
            '%(message)s\n'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Console handler for visibility
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def log_calculation(self, component: str, operation: str, 
                       inputs: Dict[str, Any], formula: str,
                       calculation: str, output: Any, interpretation: str):
        """Log a calculation with full transparency."""
        msg = f"""
Operation: {operation}
Component: {component}
Input: {inputs}
Formula: {formula}
Calculation: {calculation}
Output: {output}
Interpretation: {interpretation}
"""
        self.logger.info(msg)
    
    def log_contradiction(self, stmt1: Statement, stmt2: Statement, reason: str):
        """Log a detected contradiction."""
        msg = f"""
CONTRADICTION DETECTED
Statement 1: "{stmt1.content}" (at {stmt1.timestamp})
Statement 2: "{stmt2.content}" (at {stmt2.timestamp})
Reason: {reason}
Context sizes: {stmt1.context_size} vs {stmt2.context_size}
"""
        self.logger.warning(msg)
    
    def log_threshold_check(self, metric_name: str, value: float, threshold: float, passed: bool):
        """Log threshold checking."""
        status = "PASS" if passed else "FAIL"
        msg = f"""
Threshold Check: {metric_name}
Value: {value}
Threshold: {threshold}
Status: {status}
"""
        level = logging.INFO if passed else logging.WARNING
        self.logger.log(level, msg)


class CoherenceTracker:
    """
    Tracks logical consistency across all system assertions.

    By default this is a heuristic contradiction tracker with optional
    embedding and model-assisted paths when those dependencies are available.
    """
    
    def __init__(self, logger: AuditLogger, use_semantic: bool = True):
        self.logger = logger
        self.statements: List[Statement] = []
        self.contradictions: List[Contradiction] = []
        
        # Initialize semantic analyzer if available
        self.semantic_analyzer = None
        if use_semantic and SEMANTIC_AVAILABLE:
            try:
                self.semantic_analyzer = SemanticAnalyzer(use_embeddings=True)
                self.logger.logger.info("Semantic analyzer initialized with embeddings")
            except Exception as e:
                self.logger.logger.warning(f"Could not initialize semantic analyzer: {e}")
                self.semantic_analyzer = None
        
        # Initialize neural models if available
        self.neural_contradiction_model = None
        if use_semantic and NEURAL_AVAILABLE:
            try:
                self.neural_contradiction_model = ContradictionDetectionModel()
                # Try to load pre-trained weights if available
                try:
                    import os
                    model_path = "models/contradiction_model.pt"
                    if os.path.exists(model_path):
                        checkpoint = torch.load(model_path, map_location='cpu')
                        self.neural_contradiction_model.load_state_dict(checkpoint['model_state_dict'])
                        self.logger.logger.info("Loaded pre-trained contradiction detection model")
                except Exception:
                    self.logger.logger.debug("No pre-trained contradiction model found, using random initialization")
