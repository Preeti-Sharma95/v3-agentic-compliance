from typing import Dict, Any, List, Optional, TypedDict
import pandas as pd
from datetime import datetime
from langchain_core.messages import BaseMessage


class ComplianceState(TypedDict):
    """Enhanced state management for compliance workflow"""

    # Core data
    session_id: str
    data: pd.DataFrame
    enhanced_data: Optional[pd.DataFrame]

    # Analysis results
    dormant_results: Optional[Dict[str, Any]]
    compliance_results: Optional[Dict[str, Any]]
    non_compliant_data: Optional[pd.DataFrame]

    # Workflow state
    current_step: str
    messages: List[BaseMessage]
    error: Optional[str]
    confidence_score: float

    # Metadata
    timestamp: datetime
    agent_logs: List[Dict[str, Any]]
    notifications: List[str]

    # Results
    final_result: Optional[str]
    recommendations: List[str]

    # MCP integration
    mcp_enabled: bool
    mcp_results: Optional[Dict[str, Any]]
