"""
Example of integrating the stability framework with an existing AI system.

This shows how to wrap your AI system to automatically monitor stability.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_system_stability_framework import AIStabilityFramework
