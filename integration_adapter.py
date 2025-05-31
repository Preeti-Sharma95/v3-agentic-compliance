# integration_adapter.py
"""
Integration Adapter for RAG-Enhanced Agentic Banking Compliance System

This adapter integrates the new RAG-Agentic architecture with the existing
banking compliance system, providing backward compatibility while enabling
enhanced capabilities.
"""

import asyncio
import pandas as pd
import uuid
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
from pathlib import Path

# Import existing system components
try:
    from orchestrator import run_flow as legacy_run_flow, State
    from agents.memory_agent import MemoryAgent
    from agents.dormant_identification_agent import DormantIdentificationAgent
    from agents.compliance_agent import ComplianceAgent
    from agents.supervisor import SupervisorAgent
    from agents.notifier import NotificationAgent

    LEGACY_SYSTEM_AVAILABLE = True
except ImportError:
    LEGACY_SYSTEM_AVAILABLE = False

# Import new RAG-Agentic system
try:
    from rag_enhanced_banking_architecture import (
        HybridComplianceOrchestrator,
        ProcessingMode,
        QueryContext,
        AgentRole,
        AgentSummary,
        ProcessingResult
    )

    RAG_AGENTIC_AVAILABLE = True
except ImportError:
    RAG_AGENTIC_AVAILABLE = False

logger = logging.getLogger(__name__)


class SystemMode:
    """System operation modes"""
    LEGACY_ONLY = "legacy_only"
    RAG_AGENTIC_ONLY = "rag_agentic_only"
    HYBRID_INTEGRATION = "hybrid_integration"
    AUTO_FALLBACK = "auto_fallback"


class IntegratedBankingComplianceSystem:
    """
    Integrated system that combines legacy agents with RAG-Agentic processing
    """

    def __init__(self, system_mode: str = SystemMode.AUTO_FALLBACK):
        self.system_mode = system_mode
        self.legacy_available = LEGACY_SYSTEM_AVAILABLE
        self.rag_agentic_available = RAG_AGENTIC_AVAILABLE

        # Initialize components
        self.rag_agentic_system: Optional[HybridComplianceOrchestrator] = None
        self.legacy_memory: Optional[MemoryAgent] = None

        # Performance tracking
        self.execution_stats = {
            "total_sessions": 0,
            "legacy_executions": 0,
            "rag_agentic_executions": 0,
            "hybrid_executions": 0,
            "fallback_executions": 0,
            "average_processing_time": 0.0,
            "success_rate": 0.0
        }

        # Initialize system
        asyncio.create_task(self._initialize_system())

    async def _initialize_system(self):
        """Initialize the integrated system"""
        try:
            logger.info(f"Initializing integrated system in mode: {self.system_mode}")

            # Initialize RAG-Agentic system if available
            if self.rag_agentic_available and self.system_mode != SystemMode.LEGACY_ONLY:
                from rag_enhanced_banking_architecture import create_hybrid_compliance_system
                self.rag_agentic_system = await create_hybrid_compliance_system()
                logger.info("RAG-Agentic system initialized successfully")

            # Initialize legacy components if available
            if self.legacy_available and self.system_mode != SystemMode.RAG_AGENTIC_ONLY:
                self.legacy_memory = MemoryAgent()
                logger.info("Legacy system components initialized")

            # Auto-select best available mode
            if self.system_mode == SystemMode.AUTO_FALLBACK:
                if self.rag_agentic_available and self.legacy_available:
                    self.system_mode = SystemMode.HYBRID_INTEGRATION
                elif self.rag_agentic_available:
                    self.system_mode = SystemMode.RAG_AGENTIC_ONLY
                elif self.legacy_available:
                    self.system_mode = SystemMode.LEGACY_ONLY
                else:
                    raise RuntimeError("No compatible system components available")

            logger.info(f"System initialized with mode: {self.system_mode}")

        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            raise

    async def run_compliance_analysis(
            self,
            data: pd.DataFrame,
            session_id: Optional[str] = None,
            query: Optional[str] = None,
            processing_mode: Optional[str] = None
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Main entry point for compliance analysis

        Args:
            data: Banking data DataFrame
            session_id: Optional session identifier
            query: Optional natural language query
            processing_mode: Optional processing mode override

        Returns:
            Tuple of (final_state, agent_logs)
        """
        start_time = datetime.now()

        if not session_id:
            session_id = str(uuid.uuid4())

        try:
            # Determine processing approach
            approach = self._determine_processing_approach(data, query, processing_mode)

            logger.info(f"Processing session {session_id} with approach: {approach}")

            # Execute based on approach
            if approach == "rag_agentic":
                result_state, agent_logs = await self._run_rag_agentic_analysis(data, session_id, query)
            elif approach == "legacy":
                result_state, agent_logs = await self._run_legacy_analysis(data, session_id)
            elif approach == "hybrid":
                result_state, agent_logs = await self._run_hybrid_analysis(data, session_id, query)
            else:
                result_state, agent_logs = await self._run_fallback_analysis(data, session_id)

            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_execution_stats(approach, processing_time, True)

            # Enhance result with integration metadata
            enhanced_state = self._enhance_result_with_metadata(result_state, approach, processing_time)

            return enhanced_state, agent_logs

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Analysis failed for session {session_id}: {e}")

            # Update statistics for failure
            self._update_execution_stats("error", processing_time, False)

            # Return error state
            error_state = self._create_error_state(str(e), session_id, data.shape)
            return error_state, []

    async def _run_rag_agentic_analysis(
            self,
            data: pd.DataFrame,
            session_id: str,
            query: Optional[str] = None
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Run analysis using RAG-Agentic system"""
        try:
            if not self.rag_agentic_system:
                raise RuntimeError("RAG-Agentic system not available")

            # Default query if none provided
            if not query:
                query = "Perform comprehensive banking compliance analysis including dormancy and compliance checks"

            # Process with RAG-Agentic system
            result = await self.rag_agentic_system.process_compliance_data(
                data=data,
                query=query,
                session_id=session_id,
                processing_mode=ProcessingMode.AUTO_SELECT
            )

            # Convert to legacy format
            state_dict = self._convert_rag_result_to_legacy_format(result, data, session_id)
            agent_logs = self._extract_agent_logs_from_rag_result(result)

            return state_dict, agent_logs

        except Exception as e:
            logger.error(f"RAG-Agentic analysis failed: {e}")
            raise

    async def _run_legacy_analysis(
            self,
            data: pd.DataFrame,
            session_id: str
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Run analysis using legacy system"""
        try:
            if not self.legacy_available:
                raise RuntimeError("Legacy system not available")

            # Run legacy flow
            final_state, memory_agent = legacy_run_flow(data)

            # Convert to dictionary format
            if hasattr(final_state, '__dict__'):
                state_dict = final_state.__dict__.copy()
            else:
                state_dict = final_state

            # Ensure session_id is set
            state_dict['session_id'] = session_id

            # Extract agent logs from memory
            agent_logs = []
            if memory_agent and hasattr(memory_agent, 'memory'):
                memory_logs = memory_agent.get(session_id) or memory_agent.get(getattr(final_state, 'session_id', ''))
                agent_logs = self._convert_memory_to_agent_logs(memory_logs)

            return state_dict, agent_logs

        except Exception as e:
            logger.error(f"Legacy analysis failed: {e}")
            raise

    async def _run_hybrid_analysis(
            self,
            data: pd.DataFrame,
            session_id: str,
            query: Optional[str] = None
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Run analysis using hybrid approach combining both systems"""
        try:
            # Step 1: Run RAG-Agentic analysis for intelligent processing
            rag_state = None
            rag_logs = []

            if self.rag_agentic_system:
                try:
                    rag_state, rag_logs = await self._run_rag_agentic_analysis(data, session_id, query)
                except Exception as e:
                    logger.warning(f"RAG-Agentic processing failed in hybrid mode: {e}")

            # Step 2: Run legacy analysis for validation and completeness
            legacy_state = None
            legacy_logs = []

            if self.legacy_available:
                try:
                    legacy_state, legacy_logs = await self._run_legacy_analysis(data, f"{session_id}_legacy")
                except Exception as e:
                    logger.warning(f"Legacy processing failed in hybrid mode: {e}")

            # Step 3: Merge results intelligently
            merged_state = self._merge_analysis_results(rag_state, legacy_state, session_id, data)
            merged_logs = self._merge_agent_logs(rag_logs, legacy_logs)

            return merged_state, merged_logs

        except Exception as e:
            logger.error(f"Hybrid analysis failed: {e}")
            # Fallback to whichever system is available
            if self.rag_agentic_system:
                return await self._run_rag_agentic_analysis(data, session_id, query)
            elif self.legacy_available:
                return await self._run_legacy_analysis(data, session_id)
            else:
                raise

    async def _run_fallback_analysis(
            self,
            data: pd.DataFrame,
            session_id: str
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Run basic fallback analysis when other systems fail"""
        try:
            # Basic data analysis
            basic_analysis = {
                'session_id': session_id,
                'data': data,
                'enhanced_data': None,
                'dormant_results': self._basic_dormancy_analysis(data),
                'compliance_results': self._basic_compliance_analysis(data),
                'non_compliant_data': None,
                'current_step': 'completed',
                'messages': [],
                'error': None,
                'confidence_score': 0.6,  # Lower confidence for basic analysis
                'timestamp': datetime.now(),
                'final_result': f"Basic analysis completed for {len(data)} accounts",
                'recommendations': [
                    "Enable enhanced processing for better analysis",
                    "Review results manually for accuracy",
                    "Consider upgrading system components"
                ],
                'notifications': [
                    f"✅ Basic analysis completed for {len(data)} accounts",
                    "⚠️ Using fallback mode - limited functionality"
                ],
                'mcp_enabled': False,
                'mcp_results': None,
                'processing_mode': 'fallback',
                'system_mode': self.system_mode
            }

            # Basic agent logs
            agent_logs = [{
                'agent': 'fallback_processor',
                'timestamp': datetime.now().isoformat(),
                'status': 'success',
                'details': {
                    'message': 'Basic fallback analysis completed',
                    'accounts_processed': len(data),
                    'mode': 'fallback',
                    'capabilities': ['basic_analysis', 'data_summary']
                }
            }]

            return basic_analysis, agent_logs

        except Exception as e:
            logger.error(f"Fallback analysis failed: {e}")
            raise

    def _determine_processing_approach(
            self,
            data: pd.DataFrame,
            query: Optional[str],
            processing_mode: Optional[str]
    ) -> str:
        """Determine the best processing approach"""

        # Override mode if specified
        if processing_mode:
            if processing_mode == "rag_agentic" and self.rag_agentic_available:
                return "rag_agentic"
            elif processing_mode == "legacy" and self.legacy_available:
                return "legacy"
            elif processing_mode == "hybrid" and self.rag_agentic_available and self.legacy_available:
                return "hybrid"

        # Auto-determine based on system mode and capabilities
        if self.system_mode == SystemMode.RAG_AGENTIC_ONLY:
            return "rag_agentic" if self.rag_agentic_available else "fallback"
        elif self.system_mode == SystemMode.LEGACY_ONLY:
            return "legacy" if self.legacy_available else "fallback"
        elif self.system_mode == SystemMode.HYBRID_INTEGRATION:
            if self.rag_agentic_available and self.legacy_available:
                return "hybrid"
            elif self.rag_agentic_available:
                return "rag_agentic"
            elif self.legacy_available:
                return "legacy"
            else:
                return "fallback"
        else:
            # Auto-fallback mode - prefer RAG-Agentic, fallback to legacy, then basic
            if query and self.rag_agentic_available:
                return "rag_agentic"
            elif self.legacy_available:
                return "legacy"
            else:
                return "fallback"

    def _convert_rag_result_to_legacy_format(
            self,
            rag_result: ProcessingResult,
            data: pd.DataFrame,
            session_id: str
    ) -> Dict[str, Any]:
        """Convert RAG-Agentic result to legacy format"""
        try:
            # Extract data from RAG result
            result_data = rag_result.data

            # Create legacy-compatible state
            legacy_state = {
                'session_id': session_id,
                'data': data,
                'enhanced_data': data,  # Use original data as enhanced
                'dormant_results': None,
                'compliance_results': None,
                'non_compliant_data': None,
                'current_step': 'completed',
                'messages': [],
                'error': rag_result.error,
                'confidence_score': rag_result.confidence_score,
                'timestamp': datetime.now(),
                'final_result': str(result_data.get('response', 'RAG-Agentic analysis completed')),
                'recommendations': self._extract_recommendations_from_rag_result(result_data),
                'notifications': [
                    f"✅ RAG-Agentic analysis completed",
                    f"🎯 Confidence: {rag_result.confidence_score:.1%}",
                    f"⚡ Processing mode: {rag_result.processing_mode.value}"
                ],
                'mcp_enabled': True,
                'mcp_results': result_data,
                'processing_mode': rag_result.processing_mode.value,
                'agent_role': rag_result.agent_role.value,
                'processing_time': rag_result.processing_time
            }

            # Try to extract structured results
            if 'agentic_result' in result_data:
                agentic_data = result_data['agentic_result']
                if isinstance(agentic_data, dict) and 'result' in agentic_data:
                    legacy_state['final_result'] = agentic_data['result']

            return legacy_state

        except Exception as e:
            logger.warning(f"Failed to convert RAG result to legacy format: {e}")
            return self._create_basic_legacy_state(data, session_id, rag_result.error or str(e))

    def _extract_agent_logs_from_rag_result(self, rag_result: ProcessingResult) -> List[Dict[str, Any]]:
        """Extract agent logs from RAG result"""
        logs = []

        try:
            # Create log entry for RAG processing
            logs.append({
                'agent': f'rag_{rag_result.agent_role.value}',
                'timestamp': datetime.now().isoformat(),
                'status': rag_result.status,
                'details': {
                    'processing_mode': rag_result.processing_mode.value,
                    'confidence_score': rag_result.confidence_score,
                    'processing_time': rag_result.processing_time,
                    'memory_usage_mb': rag_result.memory_usage_mb,
                    'metadata': rag_result.metadata
                }
            })

            # Extract additional logs from result data
            result_data = rag_result.data
            if 'relevant_agents' in result_data:
                for agent in result_data['relevant_agents']:
                    logs.append({
                        'agent': f"rag_reference_{agent.get('agent_id', 'unknown')}",
                        'timestamp': datetime.now().isoformat(),
                        'status': 'referenced',
                        'details': {
                            'agent_role': agent.get('agent_role', {}).get('value', 'unknown') if hasattr(
                                agent.get('agent_role', {}), 'value') else str(agent.get('agent_role', 'unknown')),
                            'relevance_score': agent.get('relevance_score', 0),
                            'success_rate': agent.get('success_rate', 0),
                            'capabilities': agent.get('capabilities', [])
                        }
                    })

        except Exception as e:
            logger.warning(f"Failed to extract agent logs from RAG result: {e}")
            logs.append({
                'agent': 'rag_system',
                'timestamp': datetime.now().isoformat(),
                'status': 'completed',
                'details': {'note': 'RAG processing completed with limited logging'}
            })

        return logs

    def _merge_analysis_results(
            self,
            rag_state: Optional[Dict[str, Any]],
            legacy_state: Optional[Dict[str, Any]],
            session_id: str,
            data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Intelligently merge results from both systems"""
        try:
            # Initialize merged state
            merged_state = {
                'session_id': session_id,
                'data': data,
                'enhanced_data': data,
                'current_step': 'completed',
                'messages': [],
                'error': None,
                'timestamp': datetime.now(),
                'processing_mode': 'hybrid',
                'system_mode': self.system_mode
            }

            # Merge confidence scores (weighted average)
            rag_confidence = rag_state.get('confidence_score', 0.0) if rag_state else 0.0
            legacy_confidence = legacy_state.get('confidence_score', 0.0) if legacy_state else 0.0

            if rag_state and legacy_state:
                merged_state['confidence_score'] = (rag_confidence * 0.6 + legacy_confidence * 0.4)
            elif rag_state:
                merged_state['confidence_score'] = rag_confidence
            elif legacy_state:
                merged_state['confidence_score'] = legacy_confidence
            else:
                merged_state['confidence_score'] = 0.5

            # Merge results - prefer RAG results for final output
            if rag_state and rag_state.get('final_result'):
                merged_state['final_result'] = rag_state['final_result']
            elif legacy_state and legacy_state.get('result'):
                merged_state['final_result'] = legacy_state['result']
            else:
                merged_state['final_result'] = f"Hybrid analysis completed for {len(data)} accounts"

            # Merge structured data - prefer legacy for compatibility
            if legacy_state:
                merged_state.update({
                    'dormant_results': legacy_state.get('dormant_results'),
                    'compliance_results': legacy_state.get('compliance_results'),
                    'non_compliant_data': legacy_state.get('non_compliant_data')
                })

            # Merge recommendations
            recommendations = []
            if rag_state and rag_state.get('recommendations'):
                recommendations.extend(rag_state['recommendations'])
            if legacy_state and legacy_state.get('recommendations'):
                recommendations.extend(legacy_state['recommendations'])
            if not recommendations:
                recommendations = ["Hybrid analysis completed successfully"]
            merged_state['recommendations'] = recommendations

            # Merge notifications
            notifications = []
            if rag_state and rag_state.get('notifications'):
                notifications.extend(rag_state['notifications'])
            if legacy_state and legacy_state.get('notifications'):
                notifications.extend(legacy_state['notifications'])
            notifications.append("🔄 Hybrid processing: Combined RAG-Agentic and Legacy analysis")
            merged_state['notifications'] = notifications

            # Add hybrid-specific metadata
            merged_state['hybrid_metadata'] = {
                'rag_available': rag_state is not None,
                'legacy_available': legacy_state is not None,
                'rag_confidence': rag_confidence,
                'legacy_confidence': legacy_confidence,
                'merge_strategy': 'weighted_preference'
            }

            # Set MCP status
            merged_state['mcp_enabled'] = rag_state.get('mcp_enabled', False) if rag_state else False
            merged_state['mcp_results'] = rag_state.get('mcp_results') if rag_state else None

            return merged_state

        except Exception as e:
            logger.error(f"Failed to merge analysis results: {e}")
            # Return best available state
            if rag_state:
                return rag_state
            elif legacy_state:
                return legacy_state
            else:
                return self._create_basic_legacy_state(data, session_id, str(e))

    def _merge_agent_logs(
            self,
            rag_logs: List[Dict[str, Any]],
            legacy_logs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge agent logs from both systems"""
        merged_logs = []

        # Add RAG logs with prefix
        for log in rag_logs:
            log_copy = log.copy()
            if not log_copy['agent'].startswith('rag_'):
                log_copy['agent'] = f"rag_{log_copy['agent']}"
            merged_logs.append(log_copy)

        # Add legacy logs with prefix
        for log in legacy_logs:
            log_copy = log.copy()
            if not log_copy['agent'].startswith('legacy_'):
                log_copy['agent'] = f"legacy_{log_copy['agent']}"
            merged_logs.append(log_copy)

        # Add hybrid coordinator log
        merged_logs.append({
            'agent': 'hybrid_coordinator',
            'timestamp': datetime.now().isoformat(),
            'status': 'success',
            'details': {
                'message': 'Hybrid processing coordination completed',
                'rag_logs_count': len(rag_logs),
                'legacy_logs_count': len(legacy_logs),
                'merge_strategy': 'system_prefixing'
            }
        })

        return merged_logs

    def _convert_memory_to_agent_logs(self, memory_logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert legacy memory logs to agent log format"""
        agent_logs = []

        for i, log_entry in enumerate(memory_logs):
            agent_logs.append({
                'agent': f"legacy_{log_entry.get('event', f'step_{i}')}",
                'timestamp': datetime.now().isoformat(),
                'status': 'success',
                'details': log_entry.get('data', {})
            })

        return agent_logs

    def _extract_recommendations_from_rag_result(self, result_data: Dict[str, Any]) -> List[str]:
        """Extract recommendations from RAG result data"""
        recommendations = []

        # Try different possible locations for recommendations
        if 'recommendations' in result_data:
            recommendations.extend(result_data['recommendations'])

        if 'agentic_result' in result_data:
            agentic_data = result_data['agentic_result']
            if isinstance(agentic_data, dict) and 'recommendations' in agentic_data:
                recommendations.extend(agentic_data['recommendations'])

        if 'validation' in result_data:
            validation_data = result_data['validation']
            if isinstance(validation_data, dict) and 'recommendations' in validation_data:
                recommendations.extend(validation_data['recommendations'])

        # Default recommendations if none found
        if not recommendations:
            recommendations = [
                "RAG-Agentic analysis completed successfully",
                "Review detailed results for specific insights",
                "Consider enabling hybrid mode for comprehensive analysis"
            ]

        return recommendations

    def _basic_dormancy_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Basic dormancy analysis for fallback mode"""
        try:
            dormant_count = 0
            if 'Expected_Account_Dormant' in data.columns:
                dormant_count = len(
                    data[data['Expected_Account_Dormant'].astype(str).str.lower().isin(['yes', 'true', '1'])])

            return {
                'summary_kpis': {
                    'total_accounts_flagged_dormant': dormant_count,
                    'percentage_dormant_of_total': (dormant_count / len(data)) * 100 if len(data) > 0 else 0,
                    'total_dormant_balance_aed': 0,
                    'count_high_value_dormant': 0
                },
                'report_date_used': datetime.now().strftime('%Y-%m-%d'),
                'total_accounts_analyzed': len(data)
            }
        except Exception as e:
            logger.warning(f"Basic dormancy analysis failed: {e}")
            return {'summary_kpis': {'total_accounts_flagged_dormant': 0}}

    def _basic_compliance_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Basic compliance analysis for fallback mode"""
        try:
            cb_transfer_count = 0
            if 'Expected_Transfer_to_CB_Due' in data.columns:
                cb_transfer_count = len(
                    data[data['Expected_Transfer_to_CB_Due'].astype(str).str.lower().isin(['yes', 'true', '1'])])

            return {
                'total_accounts_processed': len(data),
                'transfer_candidates_cb': {'count': cb_transfer_count,
                                           'desc': f'Accounts ready for CB transfer: {cb_transfer_count}'},
                'incomplete_contact': {'count': 0, 'desc': 'Contact attempts check not available'},
                'flag_candidates': {'count': 0, 'desc': 'Flagging check not available'}
            }
        except Exception as e:
            logger.warning(f"Basic compliance analysis failed: {e}")
            return {'total_accounts_processed': len(data)}

    def _create_basic_legacy_state(self, data: pd.DataFrame, session_id: str, error: str = None) -> Dict[str, Any]:
        """Create basic legacy-compatible state"""
        return {
            'session_id': session_id,
            'data': data,
            'enhanced_data': None,
            'dormant_results': self._basic_dormancy_analysis(data),
            'compliance_results': self._basic_compliance_analysis(data),
            'non_compliant_data': None,
            'current_step': 'completed' if not error else 'error',
            'messages': [],
            'error': error,
            'confidence_score': 0.5,
            'timestamp': datetime.now(),
            'final_result': f"Basic analysis completed for {len(data)} accounts" if not error else f"Analysis failed: {error}",
            'recommendations': ["Review results manually", "Consider enabling enhanced processing"],
            'notifications': [f"✅ Basic processing completed for {len(data)} accounts"],
            'mcp_enabled': False,
            'mcp_results': None
        }

    def _create_error_state(self, error: str, session_id: str, data_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Create error state"""
        return {
            'session_id': session_id,
            'data': pd.DataFrame(),
            'enhanced_data': None,
            'dormant_results': None,
            'compliance_results': None,
            'non_compliant_data': None,
            'current_step': 'error',
            'messages': [],
            'error': error,
            'confidence_score': 0.0,
            'timestamp': datetime.now(),
            'final_result': f'Analysis failed: {error}',
            'recommendations': [
                'Check system configuration',
                'Verify data format',
                'Review system logs',
                'Try alternative processing mode'
            ],
            'notifications': [f'❌ Analysis failed: {error}'],
            'mcp_enabled': False,
            'mcp_results': None,
            'system_mode': self.system_mode,
            'data_shape': data_shape
        }

    def _enhance_result_with_metadata(self, state: Dict[str, Any], approach: str, processing_time: float) -> Dict[
        str, Any]:
        """Enhance result with integration metadata"""
        state = state.copy()

        # Add integration metadata
        state['integration_metadata'] = {
            'processing_approach': approach,
            'processing_time': processing_time,
            'system_mode': self.system_mode,
            'components_available': {
                'rag_agentic': self.rag_agentic_available,
                'legacy': self.legacy_available
            },
            'integration_version': '1.0.0',
            'timestamp': datetime.now().isoformat()
        }

        # Enhance notifications with system info
        if 'notifications' not in state:
            state['notifications'] = []

        state['notifications'].append(f"🔧 Processing approach: {approach}")
        state['notifications'].append(f"⚡ Processing time: {processing_time:.2f}s")

        return state

    def _update_execution_stats(self, approach: str, processing_time: float, success: bool):
        """Update execution statistics"""
        self.execution_stats['total_sessions'] += 1

        # Update approach-specific counters
        if approach == 'rag_agentic':
            self.execution_stats['rag_agentic_executions'] += 1
        elif approach == 'legacy':
            self.execution_stats['legacy_executions'] += 1
        elif approach == 'hybrid':
            self.execution_stats['hybrid_executions'] += 1
        else:
            self.execution_stats['fallback_executions'] += 1

        # Update average processing time
        total_sessions = self.execution_stats['total_sessions']
        current_avg = self.execution_stats['average_processing_time']
        self.execution_stats['average_processing_time'] = (
                                                                  (current_avg * (total_sessions - 1)) + processing_time
                                                          ) / total_sessions

        # Update success rate
        if success:
            current_success_rate = self.execution_stats['success_rate']
            self.execution_stats['success_rate'] = (
                                                           (current_success_rate * (total_sessions - 1)) + 1.0
                                                   ) / total_sessions
        else:
            current_success_rate = self.execution_stats['success_rate']
            self.execution_stats['success_rate'] = (
                                                           current_success_rate * (total_sessions - 1)
                                                   ) / total_sessions

    # Public API methods

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'system_mode': self.system_mode,
            'components': {
                'rag_agentic_available': self.rag_agentic_available,
                'legacy_available': self.legacy_available,
                'rag_agentic_initialized': self.rag_agentic_system is not None,
                'legacy_initialized': self.legacy_memory is not None
            },
            'execution_stats': self.execution_stats.copy(),
            'capabilities': {
                'natural_language_queries': self.rag_agentic_available,
                'legacy_compatibility': self.legacy_available,
                'hybrid_processing': self.rag_agentic_available and self.legacy_available,
                'fallback_processing': True
            }
        }

        # Add system-specific status
        if self.rag_agentic_system:
            try:
                status['rag_agentic_status'] = self.rag_agentic_system.get_system_status()
            except Exception as e:
                status['rag_agentic_status'] = {'error': str(e)}

        return status

    async def get_processing_recommendations(self, data: pd.DataFrame, query: Optional[str] = None) -> Dict[str, Any]:
        """Get recommendations for processing the given data and query"""
        try:
            recommendations = {
                'recommended_approach': self._determine_processing_approach(data, query, None),
                'data_analysis': {
                    'shape': data.shape,
                    'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024,
                    'columns': len(data.columns),
                    'has_required_columns': self._check_required_columns(data)
                },
                'system_capabilities': {
                    'rag_agentic_available': self.rag_agentic_available,
                    'legacy_available': self.legacy_available,
                    'natural_language_processing': bool(query and self.rag_agentic_available),
                    'structured_analysis': self.legacy_available
                },
                'recommendations': []
            }

            # Add specific recommendations
            if query and self.rag_agentic_available:
                recommendations['recommendations'].append(
                    "RAG-Agentic processing recommended for natural language queries")

            if data.shape[0] > 10000:
                recommendations['recommendations'].append("Large dataset detected - consider performance optimizations")

            if not self._check_required_columns(data):
                recommendations['recommendations'].append("Some required columns missing - results may be limited")

            # Get RAG-specific recommendations if available
            if self.rag_agentic_system and query:
                try:
                    rag_recommendations = await self.rag_agentic_system.get_query_recommendations(query, data)
                    recommendations['rag_specific'] = rag_recommendations
                except Exception as e:
                    logger.warning(f"Failed to get RAG recommendations: {e}")

            return recommendations

        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return {'error': str(e)}

    def _check_required_columns(self, data: pd.DataFrame) -> bool:
        """Check if data has required columns for compliance analysis"""
        required_columns = ['Account_ID']
        optional_important_columns = [
            'Expected_Account_Dormant',
            'Date_Last_Cust_Initiated_Activity',
            'Account_Type',
            'Current_Balance'
        ]

        has_required = all(col in data.columns for col in required_columns)
        has_some_optional = any(col in data.columns for col in optional_important_columns)

        return has_required and has_some_optional

    async def close(self):
        """Close and cleanup the integrated system"""
        try:
            if self.rag_agentic_system:
                # Close RAG-Agentic components if they have cleanup methods
                if hasattr(self.rag_agentic_system, 'close'):
                    await self.rag_agentic_system.close()

            logger.info("Integrated Banking Compliance System closed successfully")

        except Exception as e:
            logger.error(f"Error during system cleanup: {e}")


# Factory function for easy initialization
async def create_integrated_banking_system(
        system_mode: str = SystemMode.AUTO_FALLBACK) -> IntegratedBankingComplianceSystem:
    """Create and initialize the integrated banking compliance system"""
    system = IntegratedBankingComplianceSystem(system_mode)

    # Wait for initialization
    await asyncio.sleep(2)

    return system


# Example usage demonstrating the integration
async def example_integration_usage():
    """Example usage of the integrated system"""
    print("=== Integrated Banking Compliance System Demo ===")

    # Create integrated system
    system = await create_integrated_banking_system(SystemMode.AUTO_FALLBACK)

    # Get system status
    status = system.get_system_status()
    print(f"System mode: {status['system_mode']}")
    print(f"Components available: {status['components']}")

    # Create sample data
    sample_data = pd.DataFrame({
        'Account_ID': [f'ACC{i:06d}' for i in range(50)],
        'Account_Type': ['Current', 'Savings', 'Fixed'] * 16 + ['Investment', 'Investment'],
        'Current_Balance': pd.Series(range(50)) * 1000 + 5000,
        'Expected_Account_Dormant': ['yes' if i % 7 == 0 else 'no' for i in range(50)],
        'Date_Last_Cust_Initiated_Activity': pd.date_range('2020-01-01', periods=50, freq='30D'),
        'Expected_Transfer_to_CB_Due': ['yes' if i % 15 == 0 else 'no' for i in range(50)]
    })

    print(f"Sample data shape: {sample_data.shape}")

    # Test different processing approaches
    test_cases = [
        {
            'name': 'Basic Analysis',
            'query': None,
            'mode': None
        },
        {
            'name': 'Natural Language Query',
            'query': 'How many dormant accounts need CB transfer?',
            'mode': None
        },
        {
            'name': 'Forced Legacy Mode',
            'query': 'Analyze compliance issues',
            'mode': 'legacy'
        },
        {
            'name': 'Forced RAG-Agentic Mode',
            'query': 'Perform comprehensive dormancy analysis',
            'mode': 'rag_agentic'
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print(f"   Query: {test_case['query'] or 'None'}")
        print(f"   Mode: {test_case['mode'] or 'Auto'}")

        try:
            # Get processing recommendations
            recommendations = await system.get_processing_recommendations(
                sample_data, test_case['query']
            )
            print(f"   Recommended approach: {recommendations.get('recommended_approach', 'unknown')}")

            # Run analysis
            start_time = datetime.now()
            result_state, agent_logs = await system.run_compliance_analysis(
                data=sample_data,
                session_id=f"demo_session_{i}",
                query=test_case['query'],
                processing_mode=test_case['mode']
            )
            processing_time = (datetime.now() - start_time).total_seconds()

            # Display results
            print(f"   ✅ Status: {result_state.get('current_step', 'unknown')}")
            print(f"   🎯 Confidence: {result_state.get('confidence_score', 0):.1%}")
            print(f"   ⚡ Time: {processing_time:.2f}s")
            print(
                f"   🔧 Approach used: {result_state.get('integration_metadata', {}).get('processing_approach', 'unknown')}")
            print(f"   📊 Agent steps: {len(agent_logs)}")

            # Show key results
            if 'dormant_results' in result_state and result_state['dormant_results']:
                dormant_count = result_state['dormant_results'].get('summary_kpis', {}).get(
                    'total_accounts_flagged_dormant', 0)
                print(f"   🏦 Dormant accounts: {dormant_count}")

            if 'compliance_results' in result_state and result_state['compliance_results']:
                cb_transfer = result_state['compliance_results'].get('transfer_candidates_cb', {}).get('count', 0)
                print(f"   🚨 CB transfer needed: {cb_transfer}")

        except Exception as e:
            print(f"   ❌ Error: {e}")

    # Show final system statistics
    print(f"\n📈 Final System Statistics:")
    final_status = system.get_system_status()
    stats = final_status['execution_stats']
    print(f"   Total sessions: {stats['total_sessions']}")
    print(f"   Success rate: {stats['success_rate']:.1%}")
    print(f"   Average processing time: {stats['average_processing_time']:.2f}s")
    print(f"   RAG-Agentic executions: {stats['rag_agentic_executions']}")
    print(f"   Legacy executions: {stats['legacy_executions']}")
    print(f"   Hybrid executions: {stats['hybrid_executions']}")
    print(f"   Fallback executions: {stats['fallback_executions']}")

    # Cleanup
    await system.close()
    print("\n✅ Demo completed successfully!")


if __name__ == "__main__":
    # Run the integration demo
    asyncio.run(example_integration_usage())