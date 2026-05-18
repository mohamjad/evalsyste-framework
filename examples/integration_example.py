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
        
        # Add monitoring metadata to response
        if isinstance(response, dict):
            response['stability_metrics'] = stability_result['metrics']
            response['signal_quality'] = stability_result['signal_quality']
            response['thresholds_passed'] = stability_result['thresholds_passed']
        else:
            # If response is an object, add attributes
            response.stability_metrics = stability_result['metrics']
            response.signal_quality = stability_result['signal_quality']
            response.thresholds_passed = stability_result['thresholds_passed']
        
        return response
    
    def _extract_statements(self, response: Any) -> List[str]:
        """
        Extract statements/claims from AI response.
        
        This is a simple example - you'd customize this based on
        how your AI system structures its output.
        """
        statements = []
        
        if isinstance(response, dict):
            # If response is a dict, look for common fields
            if 'message' in response:
                statements.append(response['message'])
            if 'reasoning' in response:
                statements.append(response['reasoning'])
            if 'claims' in response:
                statements.extend(response['claims'])
        elif isinstance(response, str):
            # If response is a string, split into sentences
            statements = [s.strip() for s in response.split('.') if s.strip()]
        elif hasattr(response, 'text'):
            # If response has a text attribute
            statements = [response.text]
        else:
            # Fallback: convert to string
            statements = [str(response)]
        
        return statements
    
    def _get_context_size(self) -> int:
        """
        Get current context size from the AI system.
        
        Adjust this based on how your system tracks context.
        """
        if hasattr(self.base, 'context_window'):
            return len(self.base.context_window)
        elif hasattr(self.base, 'get_context_size'):
            return self.base.get_context_size()
        else:
            # Default estimate
            return 1000
    
    def get_stability_report(self) -> str:
        """Get current stability report."""
        return self.monitor.generate_stability_report()


# Example usage with a mock AI system
class MockAISystem:
    """Simple mock AI system for demonstration."""
    
    def __init__(self):
        self.context_window = []
        self.operation_count = 0
    
    def process(self, input_data: str) -> Dict[str, Any]:
        """Process input and return response."""
        self.operation_count += 1
        self.context_window.append(input_data)
        
        return {
            'message': f"Processed: {input_data}",
            'reasoning': f"This is operation #{self.operation_count}",
