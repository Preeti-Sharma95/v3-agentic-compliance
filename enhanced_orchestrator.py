"""
Enhanced Orchestrator - Bridge between legacy and new systems
Place this file in your project root directory (same level as main.py)
"""

import uuid
import asyncio
from typing import Tuple, Dict, Any, List
import pandas as pd
from datetime import datetime
import logging

# Import existing components
from orchestrator import run_flow as legacy_run_flow

# Import configuration
try:
    from config.config import config

    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    config = type('Config', (), {
        'langsmith_enabled': False,
        'mcp_enabled': False,
        'environment': 'development'
    })()

# Import utilities
try:
    from utils.langsmith_setup import setup_langsmith

    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False


    def setup_langsmith():
        return False

# Setup logging
logger = logging.getLogger(__name__)


class EnhancedComplianceOrchestrator:
    """Bridge orchestrator that supports both legacy and enhanced modes"""

    def __init__(self, use_enhanced: bool = False):
        # For now, default to legacy mode until LangGraph is fully implemented
        self.use_enhanced = False  # Always use legacy for now

        # Setup LangSmith if available
        if UTILS_AVAILABLE:
            setup_langsmith()

        logger.info(f"Enhanced orchestrator initialized in {'enhanced' if self.use_enhanced else 'legacy'} mode")

    def run_flow(self, data: pd.DataFrame) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Run compliance flow - currently uses legacy mode with enhanced formatting"""

        try:
            # Run legacy flow
            state, memory = legacy_run_flow(data)

            # Convert to enhanced format for UI compatibility
            enhanced_state = self._convert_legacy_state(state, memory)
            agent_logs = self._extract_agent_logs(state, memory)

            logger.info(f"Analysis completed successfully for {len(data)} accounts")
            return enhanced_state, agent_logs

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            # Return error state
            error_state = self._create_error_state(str(e))
            return error_state, []

    def _convert_legacy_state(self, legacy_state, memory) -> Dict[str, Any]:
        """Convert legacy state to enhanced format for UI compatibility"""

        session_id = getattr(legacy_state, 'session_id', str(uuid.uuid4()))

        # Extract dormant and compliance results if available
        dormant_results = getattr(legacy_state, 'dormant_results', None)
        compliance_results = getattr(legacy_state, 'compliance_results', None)

        # Calculate confidence score based on available data
        confidence_score = self._calculate_confidence_score(legacy_state)

        # Generate recommendations
        recommendations = self._generate_recommendations(legacy_state)

        # Generate notifications
        notifications = self._generate_notifications(legacy_state)

        enhanced_state = {
            "session_id": session_id,
            "data": legacy_state.data,
            "enhanced_data": getattr(legacy_state, 'enhanced_data', None),
            "dormant_results": dormant_results,
            "compliance_results": compliance_results,
            "non_compliant_data": getattr(legacy_state, 'non_compliant_data', None),
            "current_step": "completed",
            "messages": [],
            "error": getattr(legacy_state, 'error', None),
            "confidence_score": confidence_score,
            "timestamp": datetime.now(),
            "agent_logs": self._extract_agent_logs(legacy_state, memory),
            "notifications": notifications,
            "final_result": getattr(legacy_state, 'result', "Analysis completed successfully"),
            "recommendations": recommendations,
            "mcp_enabled": config.mcp_enabled if CONFIG_AVAILABLE else False,
            "mcp_results": None
        }

        return enhanced_state

    def _extract_agent_logs(self, legacy_state, memory) -> List[Dict[str, Any]]:
        """Extract agent logs from memory"""
        logs = []

        if memory and hasattr(legacy_state, 'session_id'):
            memory_logs = memory.get(legacy_state.session_id)
            if memory_logs:
                for i, log_entry in enumerate(memory_logs):
                    logs.append({
                        "agent": log_entry.get('event', f'step_{i}'),
                        "timestamp": datetime.now().isoformat(),
                        "status": "success",
                        "details": log_entry.get('data', {})
                    })

        # Add default log if no logs found
        if not logs:
            logs.append({
                "agent": "legacy_analysis",
                "timestamp": datetime.now().isoformat(),
                "status": "success",
                "details": {"message": "Legacy analysis completed"}
            })

        return logs

    def _calculate_confidence_score(self, legacy_state) -> float:
        """Calculate confidence score based on legacy state"""
        base_score = 0.8

        # Reduce confidence if there are errors
        if getattr(legacy_state, 'error', None):
            base_score -= 0.3

        # Increase confidence if we have results
        if getattr(legacy_state, 'dormant_results', None):
            base_score += 0.1

        if getattr(legacy_state, 'compliance_results', None):
            base_score += 0.1

        return min(max(base_score, 0.0), 1.0)

    def _generate_recommendations(self, legacy_state) -> List[str]:
        """Generate recommendations based on legacy state"""
        recommendations = []

        # Check for compliance issues
        compliance_results = getattr(legacy_state, 'compliance_results', None)
        if compliance_results:
            if compliance_results.get('transfer_candidates_cb', {}).get('count', 0) > 0:
                recommendations.append("URGENT: Initiate CBUAE transfer process for eligible accounts")

            if compliance_results.get('incomplete_contact', {}).get('count', 0) > 0:
                recommendations.append("Address incomplete contact attempts for dormant accounts")

        # Check for dormant accounts
        dormant_results = getattr(legacy_state, 'dormant_results', None)
        if dormant_results and dormant_results.get('summary_kpis', {}).get('count_high_value_dormant', 0) > 0:
            recommendations.append("Review high-value dormant accounts for special handling")

        # Default recommendations if none generated
        if not recommendations:
            recommendations = [
                "Continue regular compliance monitoring",
                "Review dormancy identification processes",
                "Maintain documentation for regulatory audits"
            ]

        return recommendations

    def _generate_notifications(self, legacy_state) -> List[str]:
        """Generate notifications based on legacy state"""
        notifications = []

        # Check for critical issues
        compliance_results = getattr(legacy_state, 'compliance_results', None)
        if compliance_results:
            cb_transfer_count = compliance_results.get('transfer_candidates_cb', {}).get('count', 0)
            if cb_transfer_count > 0:
                notifications.append(f"🚨 URGENT: {cb_transfer_count} accounts require CBUAE transfer")

            incomplete_contact = compliance_results.get('incomplete_contact', {}).get('count', 0)
            if incomplete_contact > 0:
                notifications.append(f"⚠️ WARNING: {incomplete_contact} accounts have incomplete contact attempts")

        # Check non-compliant data
        non_compliant_data = getattr(legacy_state, 'non_compliant_data', None)
        if non_compliant_data is not None and not non_compliant_data.empty:
            notifications.append(f"📋 INFO: {len(non_compliant_data)} non-compliant accounts identified")

        return notifications

    def _create_error_state(self, error_message: str) -> Dict[str, Any]:
        """Create error state for failed analysis"""
        return {
            "session_id": str(uuid.uuid4()),
            "data": pd.DataFrame(),
            "enhanced_data": None,
            "dormant_results": None,
            "compliance_results": None,
            "non_compliant_data": None,
            "current_step": "error",
            "messages": [],
            "error": error_message,
            "confidence_score": 0.0,
            "timestamp": datetime.now(),
            "agent_logs": [{
                "agent": "error_handler",
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "details": {"error": error_message}
            }],
            "notifications": [f"❌ Analysis failed: {error_message}"],
            "final_result": f"Analysis failed: {error_message}",
            "recommendations": ["Check system configuration", "Review input data", "Contact support if issue persists"],
            "mcp_enabled": False,
            "mcp_results": None
        }


# Backward compatibility function
def run_enhanced_flow(data: pd.DataFrame) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Backward compatibility function"""
    orchestrator = EnhancedComplianceOrchestrator()
    return orchestrator.run_flow(data)