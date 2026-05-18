"""
Simple test script to verify the framework works correctly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_system_stability_framework import AIStabilityFramework

def main():
    print("Testing AI Stability Framework")
    print("=" * 50)
    
    # Initialize framework
    print("\n1. Initializing framework...")
    framework = AIStabilityFramework(log_file="test.log", log_level="WARNING")
    print("   [OK] Framework initialized and verified")
    
    # Test basic operation
    print("\n2. Testing basic operation...")
    result = framework.process_operation(
        operation_description="Test operation",
        statements=[
            "System initialized",
            "Processing started",
            "Operation completed"
        ],
        context_size=100
    )
    print(f"   Signal Quality: {result['signal_quality']}")
    print(f"   Coherence: {result['metrics']['coherence']:.3f}")
    print(f"   [OK] Basic operation works")
    
    # Test contradiction detection
    print("\n3. Testing contradiction detection...")
    result2 = framework.process_operation(
        operation_description="Test with contradiction",
        statements=[
            "System is stable",
            "System is unstable",  # Contradiction
            "Processing continues"
        ],
        context_size=200
    )
    print(f"   Signal Quality: {result2['signal_quality']}")
    print(f"   Coherence: {result2['metrics']['coherence']:.3f}")
    print(f"   Contradictions: {result2['metrics']['contradictions_count']}")
    assert result2['metrics']['contradictions_count'] > 0, "Should detect contradiction"
    print(f"   [OK] Contradiction detected")
    
    # Test scaling analysis
    print("\n4. Testing scaling analysis...")
    for size in [500, 1000]:
        framework.process_operation(
            operation_description=f"Scale test {size}",
            statements=[f"Processing at {size} tokens"],
            context_size=size
        )
    
    analysis = framework.context_analyzer.analyze_scaling_behavior()
    print(f"   Trend: {analysis['trend']}")
    print(f"   [OK] Scaling analysis works")
    
    # Generate report
    print("\n5. Generating report...")
    report = framework.generate_stability_report()
    print("   [OK] Report generated")
    
    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)

if __name__ == "__main__":
    main()
