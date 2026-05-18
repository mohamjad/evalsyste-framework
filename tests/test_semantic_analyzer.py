"""Tests for lexical semantic-analysis fallbacks."""

from semantic_analyzer import SemanticAnalyzer


def test_lexical_contradiction_detects_antonyms():
    """Antonym patterns should trigger contradiction detection without embeddings."""
    analyzer = SemanticAnalyzer(use_embeddings=False)

    is_contradiction, confidence, reason = analyzer.detect_contradiction(
        "The system is stable",
        "The system is unstable",
    )

    assert is_contradiction is True
    assert confidence > 0.6
    assert "stable" in reason or "unstable" in reason


def test_redundancy_increases_for_duplicate_text():
    """Repeated content should look more redundant than distinct content."""
    analyzer = SemanticAnalyzer(use_embeddings=False)
