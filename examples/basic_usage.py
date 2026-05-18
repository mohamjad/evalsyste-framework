"""
Basic usage examples for the AI Stability Framework.

These examples show how to use the framework in common scenarios.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_system_stability_framework import AIStabilityFramework


def example_1_detecting_degradation():
    """Example: Detecting when context causes degradation."""
    print("=" * 60)
    print("Example 1: Detecting Context Degradation")
    print("=" * 60)
    
    framework = AIStabilityFramework(log_file="example1.log", log_level="INFO")
    
    # Small context - should be coherent
    print("\n1. Small context (100 tokens)...")
    result1 = framework.process_operation(
        operation_description="Initial query",
        statements=[
            "User wants weather information",
            "Location is London",
            "Retrieved forecast data"
        ],
        context_size=100
    )
    print(f"   Signal Quality: {result1['signal_quality']}")
    print(f"   Coherence: {result1['metrics']['coherence']:.3f}")
    
    # Medium context - still good
    print("\n2. Medium context (500 tokens)...")
    result2 = framework.process_operation(
        operation_description="Context enrichment",
        statements=[
            "User wants weather information",
            "Location is London",
            "Retrieved forecast data",
            "Temperature is 15C",
            "Weather is sunny"
        ],
        context_size=500
    )
    print(f"   Signal Quality: {result2['signal_quality']}")
    print(f"   Coherence: {result2['metrics']['coherence']:.3f}")
    print(f"   Efficiency: {result2['metrics']['context_efficiency']}")
    
    # Large context - degrading with contradictions
    print("\n3. Large context (2000 tokens) - introducing contradictions...")
    result3 = framework.process_operation(
