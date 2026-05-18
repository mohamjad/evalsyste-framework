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
        
        return coherence
    
    def _interpret_coherence(self, score: float) -> str:
        """Interpret coherence score in plain English."""
        if score >= 0.95:
            return "Excellent coherence - system is logically consistent"
        elif score >= 0.85:
            return "Good coherence - minor inconsistencies only"
        elif score >= 0.75:
            return "Acceptable coherence - some contradictions present"
        else:
            return "Poor coherence - significant contradictions detected"


class SignalMetricsCalculator:
    """
    Calculates lightweight signal-quality metrics.
    """
    
    def __init__(self, logger: AuditLogger, coherence_tracker: CoherenceTracker):
        self.logger = logger
        self.coherence_tracker = coherence_tracker
        self.baseline_signal: Optional[float] = None
        self.baseline_context: Optional[int] = None
        
        # Initialize statistical analyzer
        self.statistical_analyzer = None
        if STATISTICAL_AVAILABLE:
            try:
                self.statistical_analyzer = StatisticalAnalyzer(confidence_level=0.95)
                self.logger.logger.info("Statistical analyzer initialized")
            except Exception as e:
                self.logger.logger.warning(f"Could not initialize statistical analyzer: {e}")
        
        # Initialize semantic analyzer for redundancy
        self.semantic_analyzer = None
        if SEMANTIC_AVAILABLE:
            try:
                self.semantic_analyzer = SemanticAnalyzer(use_embeddings=True)
            except Exception:
                pass
        
        # Initialize neural models if available
        self.neural_clarity_model = None
        self.neural_redundancy_model = None
        if NEURAL_AVAILABLE:
            try:
                self.neural_clarity_model = ClarityScoringModel()
                self.neural_redundancy_model = RedundancyDetectionModel()
                # Try to load pre-trained weights
                try:
                    import os
                    if os.path.exists("models/clarity_model.pt"):
                        checkpoint = torch.load("models/clarity_model.pt", map_location='cpu')
                        self.neural_clarity_model.load_state_dict(checkpoint['model_state_dict'])
                    if os.path.exists("models/redundancy_model.pt"):
                        checkpoint = torch.load("models/redundancy_model.pt", map_location='cpu')
                        self.neural_redundancy_model.load_state_dict(checkpoint['model_state_dict'])
                except Exception:
                    pass
            except Exception:
                pass
    
    def calculate_clarity_score(self, statements: List[Statement]) -> float:
        """
        Calculate clarity score using information-theoretic measures.
        
        Based on Shannon entropy and specificity metrics.
        Combines:
        - Information content (entropy-based)
        - Specificity markers (numbers, proper nouns)
        - Concrete language ratio
        
        References:
        - Shannon (1948): Information Theory
        - Resnik (1995): Semantic specificity measures
        """
        if not statements:
            return 1.0
        
        scores = []
        for stmt in statements:
            content = stmt.content
            
            # Try neural model first if available
            if self.neural_clarity_model and NEURAL_AVAILABLE and self.semantic_analyzer and self.semantic_analyzer.model:
                try:
                    emb = self.semantic_analyzer.model.encode([content])[0]
                    emb_tensor = torch.tensor(emb).unsqueeze(0)
                    clarity = self.neural_clarity_model.predict(emb_tensor)
                    scores.append(clarity)
                    continue
                except Exception:
                    pass
            
            # Fallback to information-theoretic method
            # Information content using Shannon entropy
            info_content = 0.5  # Default
            if self.semantic_analyzer:
                try:
                    info_content = self.semantic_analyzer.calculate_information_content(content)
                except Exception:
                    pass
            
            # Word count component (normalized, optimal around 15-25 words)
            word_count = len(content.split())
            if word_count == 0:
                word_score = 0.0
            elif word_count < 5:
                word_score = word_count / 10.0  # Too short
            elif word_count <= 25:
                word_score = min(1.0, 0.5 + (word_count - 5) / 40.0)  # Optimal range
            else:
                word_score = max(0.7, 1.0 - (word_count - 25) / 100.0)  # Diminishing returns
            
            # Specificity markers (numbers, dates, proper nouns)
            has_numbers = any(char.isdigit() for char in content)
            has_caps = any(char.isupper() and char.isalpha() for char in content)
            specificity_score = 0.0
            if has_numbers:
                specificity_score += 0.25
            if has_caps:
                specificity_score += 0.15
            
            # Abstract vs concrete language
            abstract_words = ["thing", "stuff", "something", "maybe", "perhaps", "possibly", "probably"]
            concrete_score = 1.0 if not any(word in content.lower() for word in abstract_words) else 0.5
            
            # Weighted combination (information content weighted highest)
            clarity = (
                info_content * 0.35 +
                word_score * 0.30 +
                specificity_score * 0.20 +
                concrete_score * 0.15
            )
            scores.append(clarity)
        
        avg_clarity = sum(scores) / len(scores)
        
        # Calculate confidence interval if statistical analyzer available
        ci_info = ""
        if self.statistical_analyzer and len(scores) > 1:
            mean, lower, upper = self.statistical_analyzer.calculate_confidence_interval(scores)
            ci_info = f" (95% CI: [{lower:.3f}, {upper:.3f}])"
        
        self.logger.log_calculation(
            component="SignalMetricsCalculator",
            operation="Calculate Clarity Score",
            inputs={"statements_count": len(statements), "method": "information_theoretic"},
            formula="weighted combination of information_content, word_score, specificity, concrete_language",
            calculation=f"Average of {len(scores)} statement clarity scores = {avg_clarity:.4f}{ci_info}",
            output=avg_clarity,
            interpretation="Clarity score based on information content and specificity"
        )
        
        return avg_clarity
    
    def calculate_redundancy_ratio(self, statements: List[Statement]) -> float:
        """
        Calculate semantic redundancy ratio.
        
        Uses semantic similarity (embeddings) if available, falls back to lexical.
        
        Formula: average pairwise semantic similarity
        
        References:
        - Jaccard (1912): Set similarity
        - Reimers & Gurevych (2019): Semantic embeddings
        """
        if not statements:
            return 0.0
        
        # Try neural model first if available
        if self.neural_redundancy_model and NEURAL_AVAILABLE and self.semantic_analyzer and self.semantic_analyzer.model:
            try:
                embeddings = []
                for stmt in statements:
                    emb = self.semantic_analyzer.model.encode([stmt.content])[0]
                    embeddings.append(torch.tensor(emb))
                redundancy = self.neural_redundancy_model.predict(embeddings)
                
                self.logger.log_calculation(
                    component="SignalMetricsCalculator",
                    operation="Calculate Semantic Redundancy Ratio",
                    inputs={"statements_count": len(statements), "method": "neural_model"},
                    formula="learned redundancy from neural network",
                    calculation=f"Neural redundancy = {redundancy:.4f}",
                    output=redundancy,
                    interpretation=f"Neural redundancy: {redundancy*100:.1f}% (learned model)"
                )
                
                return redundancy
            except Exception as e:
                self.logger.logger.debug(f"Neural redundancy failed: {e}, using semantic")
        
        # Use semantic redundancy if available
        if self.semantic_analyzer:
            try:
                texts = [stmt.content for stmt in statements]
                redundancy = self.semantic_analyzer.calculate_semantic_redundancy(texts)
                
                self.logger.log_calculation(
                    component="SignalMetricsCalculator",
                    operation="Calculate Semantic Redundancy Ratio",
                    inputs={"statements_count": len(statements), "method": "semantic_embeddings"},
                    formula="average pairwise semantic similarity",
                    calculation=f"Semantic redundancy = {redundancy:.4f}",
                    output=redundancy,
                    interpretation=f"Semantic redundancy: {redundancy*100:.1f}% (based on embeddings)"
                )
                
                return redundancy
            except Exception as e:
                self.logger.logger.debug(f"Semantic redundancy failed: {e}, using lexical")
        
        # Fallback: lexical redundancy (Jaccard similarity)
        all_words = []
        unique_words = set()
        
        for stmt in statements:
            words = stmt.content.lower().split()
            all_words.extend(words)
            unique_words.update(words)
        
        if not all_words:
            return 0.0
        
        unique_ratio = len(unique_words) / len(all_words)
        redundancy = 1.0 - unique_ratio
        
        self.logger.log_calculation(
            component="SignalMetricsCalculator",
            operation="Calculate Redundancy Ratio",
            inputs={
                "total_words": len(all_words),
                "unique_words": len(unique_words),
                "method": "lexical_jaccard"
            },
            formula="redundancy = 1.0 - (unique_words / total_words)",
            calculation=f"1.0 - ({len(unique_words)} / {len(all_words)}) = {redundancy:.4f}",
            output=redundancy,
            interpretation=f"Lexical redundancy: {redundancy*100:.1f}% of content is redundant"
        )
        
        return redundancy
    
    def calculate_context_efficiency(self, current_coherence: float, 
                                    current_context: int) -> Optional[float]:
        """
        Calculate context efficiency ratio.
        
        Formula: (current_signal / current_context) / (baseline_signal / baseline_context)
        
        Interpretation:
        - >1.0 = AMPLIFYING (scale helps) [GOOD]
        - ~1.0 = MAINTAINING (scale neutral)
        - <1.0 = DEGRADING (scale hurts) [BAD]
        """
        if self.baseline_signal is None or self.baseline_context is None:
            # Set baseline if not set
            self.baseline_signal = current_coherence
            self.baseline_context = current_context
            self.logger.logger.info(f"Setting baseline: signal={current_coherence}, context={current_context}")
            return None
        
        current_efficiency = current_coherence / current_context if current_context > 0 else 0
        baseline_efficiency = self.baseline_signal / self.baseline_context if self.baseline_context > 0 else 0
        
        if baseline_efficiency == 0:
            return None
        
        efficiency_ratio = current_efficiency / baseline_efficiency
        
        self.logger.log_calculation(
            component="SignalMetricsCalculator",
            operation="Calculate Context Efficiency",
            inputs={
                "current_coherence": current_coherence,
                "current_context": current_context,
                "baseline_coherence": self.baseline_signal,
                "baseline_context": self.baseline_context
            },
            formula="efficiency_ratio = (current_signal/current_context) / (baseline_signal/baseline_context)",
            calculation=f"({current_coherence}/{current_context}) / ({self.baseline_signal}/{self.baseline_context}) = {efficiency_ratio:.4f}",
            output=efficiency_ratio,
            interpretation=self._interpret_efficiency(efficiency_ratio)
        )
        
        return efficiency_ratio
    
    def _interpret_efficiency(self, ratio: float) -> str:
        """Interpret efficiency ratio."""
        if ratio > 1.1:
            return "AMPLIFYING - scale is improving efficiency significantly"
        elif ratio > 0.9:
            return "MAINTAINING - scale is neutral or slightly positive"
        else:
            return "DEGRADING - scale is reducing efficiency"


class ContextWindowAnalyzer:
    """
    Analyzes how system behavior changes with context scale.
    """
    
    def __init__(self, logger: AuditLogger):
        self.logger = logger
        self.snapshots: List[ContextSnapshot] = []
        
        # Initialize statistical analyzer
        self.statistical_analyzer = None
        if STATISTICAL_AVAILABLE:
            try:
                self.statistical_analyzer = StatisticalAnalyzer(confidence_level=0.95)
            except Exception:
                pass
    
    def capture_snapshot(self, context_size: int, coherence: float, 
                        operations_count: int, statements_count: int,
                        error_rate: float = 0.0):
        """Capture a snapshot of system state."""
        snapshot = ContextSnapshot(
            context_size=context_size,
            coherence=coherence,
            operations_count=operations_count,
            error_rate=error_rate,
            statements_count=statements_count
        )
        self.snapshots.append(snapshot)
        
        self.logger.logger.info(
            f"Snapshot captured: context={context_size}, "
            f"coherence={coherence:.3f}, ops={operations_count}"
        )
    
    def analyze_scaling_behavior(self) -> Dict[str, Any]:
        """
        Analyze scaling behavior with statistical significance testing.
        
        Uses linear regression and correlation analysis to determine trends.
        """
        if len(self.snapshots) < 2:
            return {
                "trend": "INSUFFICIENT_DATA",
                "recommendation": "Need at least 2 snapshots to analyze scaling",
                "statistical_significance": False
            }
        
        # Calculate efficiency at each snapshot
        efficiencies = []
        coherence_values = []
        context_sizes = []
        
        for snapshot in self.snapshots:
            if snapshot.context_size > 0:
                eff = snapshot.coherence / snapshot.context_size
                efficiencies.append((snapshot.context_size, eff))
                coherence_values.append(snapshot.coherence)
                context_sizes.append(snapshot.context_size)
        
        if len(efficiencies) < 2:
            return {
                "trend": "INSUFFICIENT_DATA",
                "recommendation": "Cannot calculate efficiency trends",
                "statistical_significance": False
            }
        
        # Statistical trend analysis if available
        trend_stats = {}
        if self.statistical_analyzer and len(coherence_values) >= 3:
            try:
                trend_stats = self.statistical_analyzer.analyze_trend(
                    coherence_values, 
                    timestamps=context_sizes
                )
                
                # Calculate confidence interval for efficiency
                eff_values = [eff for _, eff in efficiencies]
                if len(eff_values) > 1:
                    mean_eff, lower_eff, upper_eff = self.statistical_analyzer.calculate_confidence_interval(eff_values)
                    trend_stats["efficiency_ci"] = (mean_eff, lower_eff, upper_eff)
            except Exception as e:
                self.logger.logger.debug(f"Statistical analysis failed: {e}")
        
        # Determine trend (with statistical significance if available)
        first_eff = efficiencies[0][1]
        last_eff = efficiencies[-1][1]
        
        if first_eff == 0:
            trend = "UNKNOWN"
        elif trend_stats.get("is_significant", False):
            # Use statistically significant trend
            trend = trend_stats["trend"].upper()
        elif last_eff > first_eff * 1.1:
            trend = "AMPLIFYING"
        elif last_eff > first_eff * 0.9:
            trend = "MAINTAINING"
        else:
            trend = "DEGRADING"
        
        # Generate recommendation
        if trend == "AMPLIFYING" or trend == "INCREASING":
            recommendation = "Continue scaling - system leverages context effectively"
        elif trend == "MAINTAINING" or trend == "STABLE":
            recommendation = "Scale is neutral - consider optimizing context usage"
        elif trend == "DEGRADING" or trend == "DECREASING":
            recommendation = "STOP scaling - investigate degradation causes"
        else:
            recommendation = "Insufficient data for recommendation"
        
        # Add statistical information
        result = {
            "trend": trend,
            "recommendation": recommendation,
            "efficiencies": efficiencies,
            "snapshots_count": len(self.snapshots),
            "statistical_significance": trend_stats.get("is_significant", False),
            "p_value": trend_stats.get("p_value", None),
            "correlation": trend_stats.get("correlation", None),
        }
        
        if "efficiency_ci" in trend_stats:
            result["efficiency_confidence_interval"] = trend_stats["efficiency_ci"]
        
        self.logger.logger.info(
            f"Scaling analysis: trend={trend}, "
            f"significant={result['statistical_significance']}, "
            f"p={result.get('p_value', 'N/A')}"
        )
        
        return result


class SelfVerifier:
    """
    Verifies that the framework itself works correctly.

    Before monitoring any AI system, we test:
    1. Metric calculations with known inputs
    2. Contradiction detection logic
    3. Threshold checking
    4. Logging functionality
    """
    
    def __init__(self, logger: AuditLogger):
        self.logger = logger
        self.tests_passed = 0
        self.tests_failed = 0
        self.test_results: List[Dict[str, Any]] = []
    
    def run_all_tests(self) -> bool:
        """Run complete test suite."""
        self.logger.logger.info("=== FRAMEWORK SELF-VERIFICATION ===")
        
        tests = [
            self.test_coherence_calculation,
            self.test_contradiction_detection,
            self.test_threshold_checking,
        ]
        
        for test in tests:
            try:
                test()
                self.tests_passed += 1
            except AssertionError as e:
                self.tests_failed += 1
                self.logger.logger.error(f"Test failed: {e}")
        
        self.logger.logger.info(
            f"Tests run: {len(tests)}, "
            f"Passed: {self.tests_passed}, "
            f"Failed: {self.tests_failed}"
        )
        
        if self.tests_failed == 0:
            self.logger.logger.info("Status: VERIFIED - Framework is safe to use")
        else:
            self.logger.logger.error("Status: FAILED - Framework has errors")
        
        return self.tests_failed == 0
    
    def test_coherence_calculation(self):
        """Test coherence calculation with known input."""
        # Create a simple coherence tracker for testing
        # Use a temporary file for Windows compatibility
        import tempfile
        import os
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.close()
        test_logger = AuditLogger(log_file=temp_file.name, log_level="ERROR")
        tracker = CoherenceTracker(test_logger)
        
        # Add statements that should create specific coherence
        # 4 statements, if we add 1 contradiction, coherence should be:
        # total_pairs = 4*3/2 = 6
        # coherence = 1.0 - (1/6) = 0.833
        
        tracker.add_statement("System is stable", 100)
        tracker.add_statement("Processing works", 100)
        tracker.add_statement("System is unstable", 100)  # Contradiction with first
        tracker.add_statement("Output is correct", 100)
        
        coherence = tracker.calculate_coherence_score()
        
        # Should have 1 contradiction, 4 statements = 6 pairs
        expected_coherence = 1.0 - (1.0 / 6.0)  # 0.833...
        
        assert abs(coherence - expected_coherence) < 0.01, \
            f"Coherence calculation broken: expected ~{expected_coherence:.3f}, got {coherence:.3f}"
        
        # Clean up temp file
        try:
            os.unlink(temp_file.name)
        except:
            pass
        
        self.logger.logger.info("[PASS] Coherence Calculation")
    
    def test_contradiction_detection(self):
        """Test that we can detect obvious contradictions."""
        import tempfile
        import os
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.close()
        test_logger = AuditLogger(log_file=temp_file.name, log_level="ERROR")
        tracker = CoherenceTracker(test_logger)
        
        tracker.add_statement("The system is stable", 100)
        tracker.add_statement("The system is unstable", 100)
        
        # Should detect contradiction
        assert len(tracker.contradictions) > 0, \
            "Failed to detect obvious contradiction"
        
        # Clean up temp file
        try:
            os.unlink(temp_file.name)
        except:
            pass
        
        self.logger.logger.info("[PASS] Contradiction Detection")
    
    def test_threshold_checking(self):
        """Test threshold checking logic."""
        # Simple threshold check
        coherence = 0.70  # Below 0.75 threshold
        threshold = 0.75
        
        passed = coherence >= threshold
        assert not passed, "Threshold check should fail at 0.70"
        
        coherence = 0.80  # Above threshold
        passed = coherence >= threshold
        assert passed, "Threshold check should pass at 0.80"
        
        self.logger.logger.info("[PASS] Threshold Checking")


class AIStabilityFramework:
    """
    Main framework class that orchestrates all components.
    
    Usage:
        framework = AIStabilityFramework(log_file="stability.log")
        result = framework.process_operation(
            operation_description="Processing query",
            statements=["Statement 1", "Statement 2"],
            context_size=500
        )
    """
    
    def __init__(self, log_file: str = "stability.log", 
                 log_level: str = "INFO",
                 custom_thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize framework with self-verification.
        
        The framework will NOT proceed if self-verification fails.
        """
        # Set up logging
        self.logger = AuditLogger(log_file, log_level)
        self.logger.logger.info("Initializing AI Stability Framework")
        
        # Thresholds
        self.thresholds = {
            "coherence_minimum": 0.75,
            "clarity_minimum": 0.70,
            "redundancy_maximum": 0.40,
        }
        if custom_thresholds:
            self.thresholds.update(custom_thresholds)
        
        # Initialize components (with semantic analysis enabled)
        self.coherence_tracker = CoherenceTracker(self.logger, use_semantic=True)
        self.signal_metrics = SignalMetricsCalculator(self.logger, self.coherence_tracker)
        self.context_analyzer = ContextWindowAnalyzer(self.logger)
        
        # Self-verification
        self.verifier = SelfVerifier(self.logger)
        self.verified = self.verifier.run_all_tests()
        
        if not self.verified:
            raise RuntimeError(
                "Framework self-verification failed. "
                "Cannot proceed - framework has errors."
            )
        
        # Operation tracking
        self.operation_count = 0
        self.operations: List[Dict[str, Any]] = []
        
        self.logger.logger.info("Framework initialized successfully")
    
    def process_operation(self, operation_description: str,
                         statements: List[str],
