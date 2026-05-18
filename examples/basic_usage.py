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
        operation_description="Extended processing",
        statements=[
            "User wants weather information",
            "Location is London",
            "Retrieved forecast data",
            "Temperature is 15C",
            "Actually temperature is 20C",  # CONTRADICTION!
            "Weather is sunny",
            "Weather is rainy"  # ANOTHER CONTRADICTION!
        ],
        context_size=2000
    )
    print(f"   Signal Quality: {result3['signal_quality']}")
    print(f"   Coherence: {result3['metrics']['coherence']:.3f}")
    print(f"   Contradictions: {result3['metrics']['contradictions_count']}")
    
    print("\n" + framework.generate_stability_report())


def example_2_scaling_validation():
    """Example: Validating that adding context actually helps."""
    print("\n" + "=" * 60)
    print("Example 2: Validating Scale Benefits")
    print("=" * 60)
    
    framework = AIStabilityFramework(log_file="example2.log", log_level="INFO")
    
    # Baseline
    print("\nSetting baseline...")
    framework.process_operation(
        operation_description="Baseline",
        statements=["Processing started"],
        context_size=100
    )
    
    # Add context gradually and monitor
    print("\nExpanding context gradually...")
    for context_size in [500, 1000, 2000, 5000]:
        result = framework.process_operation(
            operation_description=f"Context expansion to {context_size}",
            statements=[
                f"Analyzed {context_size} tokens",
                "Found relevant patterns",
                "Generated insights",
                "Quality maintained"
            ],
            context_size=context_size
        )
        
        efficiency = result['metrics']['context_efficiency']
        if efficiency:
            print(f"   Context {context_size}: efficiency={efficiency:.3f}")
        else:
            print(f"   Context {context_size}: baseline established")
    
    # Generate analysis
    analysis = framework.context_analyzer.analyze_scaling_behavior()
    print(f"\nScaling Trend: {analysis['trend']}")
    print(f"Recommendation: {analysis['recommendation']}")


def example_3_custom_thresholds():
    """Example: Using custom thresholds."""
    print("\n" + "=" * 60)
    print("Example 3: Custom Thresholds")
    print("=" * 60)
    
    # Stricter thresholds
    custom_config = {
        'coherence_minimum': 0.80,  # Stricter than default 0.75
        'clarity_minimum': 0.75,     # Stricter than default 0.70
        'redundancy_maximum': 0.35    # Stricter than default 0.40
    }
    
    framework = AIStabilityFramework(
        log_file="example3.log",
        log_level="INFO",
        custom_thresholds=custom_config
    )
    
    print(f"Using custom thresholds:")
    print(f"  Coherence minimum: {framework.thresholds['coherence_minimum']}")
    print(f"  Clarity minimum: {framework.thresholds['clarity_minimum']}")
    print(f"  Redundancy maximum: {framework.thresholds['redundancy_maximum']}")
    
    # Test with borderline case
    result = framework.process_operation(
        operation_description="Test with custom thresholds",
        statements=[
            "System is processing",
            "Quality is acceptable",
            "Output generated"
        ],
        context_size=500
    )
    
    print(f"\nResults:")
    print(f"  Coherence: {result['metrics']['coherence']:.3f} "
          f"({'PASS' if result['thresholds_passed']['coherence'] else 'FAIL'})")
    print(f"  Clarity: {result['metrics']['clarity']:.3f} "
          f"({'PASS' if result['thresholds_passed']['clarity'] else 'FAIL'})")
    print(f"  Redundancy: {result['metrics']['redundancy']:.3f} "
          f"({'PASS' if result['thresholds_passed']['redundancy'] else 'FAIL'})")


def example_4_continuous_monitoring():
    """Example: Continuous monitoring of a long-running process."""
    print("\n" + "=" * 60)
    print("Example 4: Continuous Monitoring")
    print("=" * 60)
    
    framework = AIStabilityFramework(log_file="example4.log", log_level="INFO")
    
    # Simulate a long-running process
    print("\nSimulating 10 operations...")
    degradation_detected = False
    
    for operation_num in range(10):
        # Simulate AI system doing something
        statements = [
            f"Processing operation {operation_num}",
            "Retrieved relevant data",
            "Generated response"
        ]
        
        # Introduce contradictions after operation 5
        if operation_num > 5:
            statements.append("System is stable")
            statements.append("System is unstable")  # Contradiction
        
        context_size = 100 + (operation_num * 100)
        
        result = framework.process_operation(
            operation_description=f"Operation {operation_num}",
            statements=statements,
            context_size=context_size
        )
        
        # Check for issues
        if result['signal_quality'] in ['DEGRADING', 'NOISY']:
            if not degradation_detected:
                print(f"\nWARNING: Quality issue detected at operation {operation_num}")
                degradation_detected = True
            print(f"   Signal Quality: {result['signal_quality']}")
            print(f"   Coherence: {result['metrics']['coherence']:.3f}")
            print(f"   Contradictions: {result['metrics']['contradictions_count']}")
    
    print("\n" + framework.generate_stability_report())


if __name__ == "__main__":
    # Run all examples
    example_1_detecting_degradation()
    example_2_scaling_validation()
    example_3_custom_thresholds()
    example_4_continuous_monitoring()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("Check the log files for detailed traces.")
    print("=" * 60)
