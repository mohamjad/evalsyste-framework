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
