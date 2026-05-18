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
            except Exception as e:
                self.logger.logger.warning(f"Could not initialize neural contradiction model: {e}")
                self.neural_contradiction_model = None
    
    def add_statement(self, content: str, context_size: int, 
                     confidence: float = 1.0, operation_id: Optional[str] = None) -> Statement:
        """Add a new statement and check for contradictions."""
        stmt = Statement(
            content=content,
            timestamp=time.time(),
            context_size=context_size,
            confidence=confidence,
            operation_id=operation_id
        )
        
        self.logger.logger.debug(f"Adding statement: '{content}'")
        
        # Check against existing statements
        for existing in self.statements:
            is_contradiction, reason = self._check_contradiction(stmt, existing)
            if is_contradiction:
                contradiction = Contradiction(
                    statement1=existing,
                    statement2=stmt,
                    reason=reason,
                    detected_at=time.time()
                )
                self.contradictions.append(contradiction)
                self.logger.log_contradiction(existing, stmt, reason)
        
        self.statements.append(stmt)
        return stmt
    
    def _check_contradiction(self, stmt1: Statement, stmt2: Statement) -> Tuple[bool, str]:
        """
        Check if two statements contradict each other.
        
        Uses multi-method approach:
        1. Semantic analysis (embeddings + lexical patterns) if available
        2. Fallback to lexical pattern matching
        """
        # Try neural model first if available
        if self.neural_contradiction_model and NEURAL_AVAILABLE and SEMANTIC_AVAILABLE:
            try:
                # Get embeddings from semantic analyzer
                if self.semantic_analyzer and self.semantic_analyzer.model:
                    emb1 = self.semantic_analyzer.model.encode([stmt1.content])[0]
                    emb2 = self.semantic_analyzer.model.encode([stmt2.content])[0]
                    
                    emb1_tensor = torch.tensor(emb1).unsqueeze(0)
                    emb2_tensor = torch.tensor(emb2).unsqueeze(0)
                    
                    is_contradiction, confidence = self.neural_contradiction_model.predict(
                        emb1_tensor, emb2_tensor, threshold=0.5
                    )
                    if is_contradiction:
                        return True, f"neural model prediction (confidence: {confidence:.2f})"
            except Exception as e:
                self.logger.logger.debug(f"Neural model failed: {e}, falling back to semantic")
        
        # Try semantic analysis if available
        if self.semantic_analyzer:
            try:
                is_contradiction, confidence, reason = self.semantic_analyzer.detect_contradiction(
                    stmt1.content, stmt2.content
                )
                if is_contradiction:
                    return True, f"{reason} (confidence: {confidence:.2f})"
            except Exception as e:
                self.logger.logger.debug(f"Semantic analysis failed: {e}, falling back to lexical")
        
        # Fallback: lexical pattern matching
        content1 = stmt1.content.lower()
        content2 = stmt2.content.lower()
        
        # Antonym patterns
        opposites = [
            ("is stable", "is unstable"), ("is stable", "is not stable"),
            ("increased", "decreased"), ("increased", "reduced"),
            ("true", "false"), ("succeeded", "failed"),
            ("working", "broken"), ("correct", "incorrect"),
        ]
        
        for opp1, opp2 in opposites:
            if opp1 in content1 and opp2 in content2:
                return True, f"lexical antonym: '{opp1}' vs '{opp2}'"
            if opp2 in content1 and opp1 in content2:
                return True, f"lexical antonym: '{opp2}' vs '{opp1}'"
        
        # Explicit negation
        if "not " + content1.replace("is ", "").replace("are ", "") in content2:
            return True, "explicit negation detected"
        if "not " + content2.replace("is ", "").replace("are ", "") in content1:
            return True, "explicit negation detected"
        
        return False, ""
    
    def calculate_coherence_score(self) -> float:
        """
        Calculate coherence score with complete transparency.
        
        Formula:
        coherence = 1.0 - (contradictions / total_possible_pairs)
        
        Where:
        - total_possible_pairs = n*(n-1)/2 for n statements
        - contradictions = number of logical conflicts detected
        
        Interpretation:
        - 1.0 = Perfect (no contradictions)
        - 0.95+ = Excellent
        - 0.85-0.95 = Good
        - 0.75-0.85 = Acceptable (our threshold)
        - <0.75 = Poor
        """
        n = len(self.statements)
        
        self.logger.logger.debug(f"Calculating coherence for {n} statements")
        
        if n <= 1:
            self.logger.logger.debug("Only 1 statement, no contradictions possible")
            return 1.0
        
        # Calculate pairs
        total_pairs = n * (n - 1) // 2
        contradictions_count = len(self.contradictions)
        
        # Calculate score
        coherence = 1.0 - (contradictions_count / total_pairs) if total_pairs > 0 else 1.0
        
        # Log the calculation
        self.logger.log_calculation(
            component="CoherenceTracker",
            operation="Calculate Coherence Score",
            inputs={
                "statements_count": n,
                "contradictions_count": contradictions_count
            },
            formula="coherence = 1.0 - (contradictions / total_possible_pairs)",
            calculation=f"1.0 - ({contradictions_count} / {total_pairs}) = {coherence:.4f}",
            output=coherence,
            interpretation=self._interpret_coherence(coherence)
        )
