"""
Example of integrating the stability framework with an existing AI system.

This shows how to wrap your AI system to automatically monitor stability.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_system_stability_framework import AIStabilityFramework
from datetime import datetime
from typing import List, Dict, Any


class MonitoredAISystem:
    """
    Wrapper that adds stability monitoring to any AI system.
    
    Usage:
        base_system = YourAISystem()
        monitored = MonitoredAISystem(base_system)
        response = monitored.process(input_data)
        # response.stability_metrics contains monitoring data
    """
    
    def __init__(self, base_ai_system, log_file: str = None):
        self.base = base_ai_system
        self.monitor = AIStabilityFramework(
            log_file=log_file or f"ai_stability_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            log_level="INFO"
        )
    
    def process(self, input_data: Any) -> Dict[str, Any]:
        """
        Process input and monitor stability.
        
        Returns response with added stability_metrics field.
        """
        # Get AI response
        response = self.base.process(input_data)
        
        # Extract statements from response
        statements = self._extract_statements(response)
        
        # Get context size (adjust based on your system)
        context_size = self._get_context_size()
        
        # Monitor stability
        stability_result = self.monitor.process_operation(
            operation_description=f"Processing: {str(input_data)[:50]}",
            statements=statements,
            context_size=context_size
        )
        
