"""Tests for lexical semantic-analysis fallbacks."""

from semantic_analyzer import SemanticAnalyzer


def test_lexical_contradiction_detects_antonyms():
    """Antonym patterns should trigger contradiction detection without embeddings."""
    analyzer = SemanticAnalyzer(use_embeddings=False)

    is_contradiction, confidence, reason = analyzer.detect_contradiction(
        "The system is stable",
