# enhanced_orchestrator_with_memory.py
"""
Enhanced Orchestrator with Hybrid Memory Agent Integration
Provides intelligent memory management and MCP-powered compliance analysis
"""

import uuid
import asyncio
from typing import Tuple, Dict, Any, List, Optional
import pandas as pd
from datetime import datetime
import logging

# Import existing components
from orchestrator import run_flow as legacy_run_flow

# Import new hybrid memory system
from agents.hybrid_memory_agent import HybridMemoryAgent
from mcp.enhanced_mcp_integration import (
    EnhancedMCPComplianceTools,
    MCPKnowledgeManager,
    create_enhanced_mcp_tools
)

# Import configuration
try:
    from config.config import config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    config = type('Config', (), {
        'langsmith_enabled': False,
        'mcp_enabled': False,
        'environment': 'development',
        'mcp_server_url': 'http://localhost:8000'
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


class MemoryEnhancedState:
    """Enhanced state with memory integration"""

    def __init__(self, session_id: str, data: pd.DataFrame, memory_agent: HybridMemoryAgent):
        self.session_id = session_id
        self.data = data
        self.memory_agent = memory_agent
        self.step = 'data_ingest'
        self.result = None
        self.error = None

        # Enhanced attributes
        self.enhanced_data = None
        self.dormant_results = None
        self.compliance_results = None
        self.non_compliant_data = None
        self.confidence_score = 0.8
        self.mcp_insights = None
        self.memory_insights = None


class MemoryEnhancedComplianceOrchestrator:
    """
    Advanced orchestrator with hybrid memory and MCP integration
    """

    def __init__(self, use_enhanced: bool = True, mcp_server_url: str = None):
        self.use_enhanced = use_enhanced
        self.mcp_server_url = mcp_server_url or getattr(config, 'mcp_server_url', 'http://localhost:8000')

        # Initialize components
        self.memory_agent: Optional[HybridMemoryAgent] = None
        self.mcp_tools: Optional[EnhancedMCPComplianceTools] = None
        self.knowledge_manager: Optional[MCPKnowledgeManager] = None

        # Performance tracking
        self.orchestrator_stats = {
            'sessions_processed': 0,
            'total_accounts_analyzed': 0,
            'average_processing_time': 0,
            'memory_operations': 0,
            'mcp_operations': 0
        }

        # Setup components
        asyncio.create_task(self._initialize_async_components())

        # Setup LangSmith if available
        if UTILS_AVAILABLE:
            setup_langsmith()

        logger.info(f"Memory-enhanced orchestrator initialized (Enhanced: {self.use_enhanced})")

    async def _initialize_async_components(self):
        """Initialize async components"""
        try:
            # Initialize hybrid memory agent
            self.memory_agent = HybridMemoryAgent(
                mcp_server_url=self.mcp_server_url,
                knowledge_cache_size=1000,
                session_cache_size=500
            )

            # Initialize MCP tools if enabled
            if getattr(config, 'mcp_enabled', False):
                self.mcp_tools = await create_enhanced_mcp_tools(self.mcp_server_url)
                if self.mcp_tools:
                    self.knowledge_manager = MCPKnowledgeManager(self.mcp_tools)
                    logger.info("MCP integration enabled and initialized")
                else:
                    logger.warning("MCP tools failed to initialize")

            logger.info("Async components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize async components: {e}")

    async def run_flow_enhanced(self, data: pd.DataFrame) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Run enhanced compliance flow with memory and MCP integration
        """
        session_id = str(uuid.uuid4())
        start_time = datetime.now()

        try:
            # Ensure components are initialized
            if not self.memory_agent:
                await self._initialize_async_components()

            # Log session start
            self.memory_agent.log(
                session_id,
                'session_start',
                {
                    'total_accounts': len(data),
                    'columns': data.columns.tolist(),
                    'enhanced_mode': self.use_enhanced
                },
                tags=['session', 'start'],
                importance=0.9
            )

            # Enhanced workflow
            if self.use_enhanced:
                enhanced_state, agent_logs = await self._run_enhanced_workflow(data, session_id)
            else:
                enhanced_state, agent_logs = await self._run_legacy_workflow(data, session_id)

            # Generate insights using memory and MCP
            enhanced_state = await self._generate_enhanced_insights(enhanced_state, session_id)

            # Calculate final metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_orchestrator_stats(len(data), processing_time)

            # Log session completion
            self.memory_agent.log(
                session_id,
                'session_complete',
                {
                    'processing_time_seconds': processing_time,
                    'confidence_score': enhanced_state.get('confidence_score', 0.8),
                    'accounts_processed': len(data)
                },
                tags=['session', 'complete'],
                importance=1.0
            )

            logger.info(f"Enhanced analysis completed for session {session_id} in {processing_time:.2f}s")
            return self._convert_to_ui_format(enhanced_state), agent_logs

        except Exception as e:
            logger.error(f"Enhanced flow failed for session {session_id}: {str(e)}")

            # Log error
            if self.memory_agent:
                self.memory_agent.log(
                    session_id,
                    'session_error',
                    {'error': str(e)},
                    tags=['session', 'error'],
                    importance=1.0
                )

            return self._create_error_state(str(e), session_id), []

    async def _run_enhanced_workflow(self, data: pd.DataFrame, session_id: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Run the enhanced workflow with memory and MCP integration
        """
        # Create enhanced state
        state = MemoryEnhancedState(session_id, data, self.memory_agent)
        agent_logs = []

        # Step 1: Enhanced Data Processing
        await self._enhanced_data_processing(state, agent_logs)

        # Step 2: Memory-Enhanced Dormant Identification
        await self._memory_enhanced_dormant_identification(state, agent_logs)

        # Step 3: MCP-Powered Compliance Analysis
        await self._mcp_powered_compliance_analysis(state, agent_logs)

        # Step 4: Intelligent Risk Assessment
        await self._intelligent_risk_assessment(state, agent_logs)

        # Step 5: Memory-Based Recommendations
        await self._memory_based_recommendations(state, agent_logs)

        return state.__dict__, agent_logs

    async def _run_legacy_workflow(self, data: pd.DataFrame, session_id: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Run legacy workflow with memory enhancement
        """
        # Run legacy flow
        legacy_state, legacy_memory = legacy_run_flow(data)

        # Convert to enhanced format with memory integration
        enhanced_state = self._enhance_legacy_state(legacy_state, legacy_memory, session_id)
        agent_logs = self._extract_enhanced_agent_logs(legacy_state, legacy_memory)

        return enhanced_state, agent_logs

    async def _enhanced_data_processing(self, state: MemoryEnhancedState, agent_logs: List[Dict[str, Any]]):
        """
        Enhanced data processing with memory context
        """
        try:
            # Log data processing start
            state.memory_agent.log(
                state.session_id,
                'data_processing_start',
                {
                    'rows': len(state.data),
                    'columns': len(state.data.columns),
                    'memory_size_mb': state.data.memory_usage(deep=True).sum() / 1024 / 1024
                },
                tags=['data', 'processing'],
                importance=0.7
            )

            # Enhanced schema validation with historical context
            historical_schemas = await self._get_historical_schemas(state.session_id)
            schema_validation = self._validate_schema_with_history(state.data, historical_schemas)

            # Data quality assessment using memory patterns
            quality_assessment = await self._assess_data_quality_with_memory(state.data, state.session_id)

            # Apply data enhancements
            state.enhanced_data = self._apply_data_enhancements(state.data, quality_assessment)

            # Log processing results
            state.memory_agent.log(
                state.session_id,
                'data_processing_complete',
                {
                    'schema_validation': schema_validation,
                    'quality_assessment': quality_assessment,
                    'enhancements_applied': True
                },
                tags=['data', 'processing', 'complete'],
                importance=0.8
            )

            agent_logs.append({
                'agent': 'enhanced_data_processor',
                'timestamp': datetime.now().isoformat(),
                'status': 'success',
                'details': {
                    'schema_valid': schema_validation.get('valid', True),
                    'quality_score': quality_assessment.get('overall_score', 0.8),
                    'enhancements': quality_assessment.get('enhancements', [])
                }
            })

            state.step = 'dormant_identification'

        except Exception as e:
            logger.error(f"Enhanced data processing failed: {e}")
            state.error = str(e)
            state.step = 'error'

    async def _memory_enhanced_dormant_identification(self, state: MemoryEnhancedState, agent_logs: List[Dict[str, Any]]):
        """
        Dormant identification enhanced with memory patterns
        """
        try:
            # Import dormant analysis
            from agents.dormant import run_all_dormant_identification_checks

            # Get historical dormancy patterns
            historical_patterns = await self._get_historical_dormancy_patterns(state.session_id)

            # Run enhanced dormant identification
            dormant_results = run_all_dormant_identification_checks(
                state.enhanced_data or state.data,
                report_date_str=datetime.now().strftime("%Y-%m-%d"),
                dormant_flags_history_df=historical_patterns
            )

            # Enhance with MCP intelligence if available
            if self.mcp_tools:
                dormant_accounts = self._extract_dormant_accounts(dormant_results)
                mcp_analysis = await self.mcp_tools.analyze_dormancy_patterns(
                    dormant_accounts, state.session_id
                )
                if mcp_analysis.success:
                    dormant_results['mcp_analysis'] = mcp_analysis.data

            state.dormant_results = dormant_results

            # Log results
            state.memory_agent.log(
                state.session_id,
                'dormant_identification_complete',
                dormant_results['summary_kpis'],
                tags=['dormancy', 'identification', 'complete'],
                importance=0.9
            )

            agent_logs.append({
                'agent': 'memory_enhanced_dormant_identifier',
                'timestamp': datetime.now().isoformat(),
                'status': 'success',
                'details': {
                    'total_dormant': dormant_results['summary_kpis']['total_accounts_flagged_dormant'],
                    'high_value_dormant': dormant_results['summary_kpis']['count_high_value_dormant'],
                    'mcp_enhanced': self.mcp_tools is not None
                }
            })

            state.step = 'compliance_analysis'

        except Exception as e:
            logger.error(f"Memory-enhanced dormant identification failed: {e}")
            state.error = str(e)
            state.step = 'error'

    async def _mcp_powered_compliance_analysis(self, state: MemoryEnhancedState, agent_logs: List[Dict[str, Any]]):
        """
        Compliance analysis powered by MCP intelligence
        """
        try:
            # Import compliance analysis
            from agents.compliance import run_all_compliance_checks

            # Run standard compliance checks
            general_threshold_date = datetime.now() - pd.Timedelta(days=3*365)
            freeze_threshold_date = datetime.now() - pd.Timedelta(days=3*365)

            compliance_results = run_all_compliance_checks(
                state.enhanced_data or state.data,
                general_threshold_date=general_threshold_date,
                freeze_threshold_date=freeze_threshold_date,
                agent_name="MCPEnhancedComplianceSystem"
            )

            # Enhance with MCP validation if available
            if self.mcp_tools:
                enhanced_compliance = await self._enhance_compliance_with_mcp(
                    compliance_results, state.session_id
                )
                compliance_results.update(enhanced_compliance)

            state.compliance_results = compliance_results

            # Extract non-compliant data
            non_compliant_items = []
            for check_name, check_result in compliance_results.items():
                if isinstance(check_result, dict) and 'df' in check_result and 'count' in check_result:
                    if check_result['count'] > 0 and not check_result['df'].empty:
                        non_compliant_items.append(check_result['df'])

            if non_compliant_items:
                state.non_compliant_data = pd.concat(non_compliant_items, ignore_index=True).drop_duplicates(
                    subset=['Account_ID'])

            # Log compliance results
            state.memory_agent.log(
                state.session_id,
                'compliance_analysis_complete',
                {
                    'total_processed': compliance_results.get('total_accounts_processed', 0),
                    'non_compliant_count': len(state.non_compliant_data) if state.non_compliant_data is not None else 0,
                    'mcp_enhanced': self.mcp_tools is not None
                },
                tags=['compliance', 'analysis', 'complete'],
                importance=0.9
            )

            agent_logs.append({
                'agent': 'mcp_powered_compliance_analyzer',
                'timestamp': datetime.now().isoformat(),
                'status': 'success',
                'details': {
                    'accounts_processed': compliance_results.get('total_accounts_processed', 0),
                    'issues_found': sum([
                        compliance_results.get('incomplete_contact', {}).get('count', 0),
                        compliance_results.get('flag_candidates', {}).get('count', 0),
                        compliance_results.get('transfer_candidates_cb', {}).get('count', 0)
                    ]),
                    'mcp_validation': self.mcp_tools is not None
                }
            })

            state.step = 'risk_assessment'

        except Exception as e:
            logger.error(f"MCP-powered compliance analysis failed: {e}")
            state.error = str(e)
            state.step = 'error'

    async def _intelligent_risk_assessment(self, state: MemoryEnhancedState, agent_logs: List[Dict[str, Any]]):
        """
        Intelligent risk assessment using memory and MCP
        """
        try:
            # Calculate base confidence score
            base_confidence = self._calculate_base_confidence(state)

            # Enhance with memory patterns
            memory_confidence = await self._calculate_memory_confidence(state.session_id)

            # Enhance with MCP insights if available
            mcp_confidence = 0.8
            if self.mcp_tools:
                mcp_confidence = await self._calculate_mcp_confidence(state)

            # Combine confidence scores
            state.confidence_score = (base_confidence * 0.4 + memory_confidence * 0.3 + mcp_confidence * 0.3)

            # Generate risk indicators
            risk_indicators = await self._generate_risk_indicators(state)

            # Log risk assessment
            state.memory_agent.log(
                state.session_id,
                'risk_assessment_complete',
                {
                    'confidence_score': state.confidence_score,
                    'base_confidence': base_confidence,
                    'memory_confidence': memory_confidence,
                    'mcp_confidence': mcp_confidence,
                    'risk_indicators': risk_indicators
                },
                tags=['risk', 'assessment', 'complete'],
                importance=0.9
            )

            agent_logs.append({
                'agent': 'intelligent_risk_assessor',
                'timestamp': datetime.now().isoformat(),
                'status': 'success',
                'details': {
                    'confidence_score': round(state.confidence_score, 3),
                    'risk_level': 'Low' if state.confidence_score >= 0.8 else 'Medium' if state.confidence_score >= 0.6 else 'High',
                    'indicators_count': len(risk_indicators),
                    'memory_enhanced': True,
                    'mcp_enhanced': self.mcp_tools is not None
                }
            })

            state.step = 'recommendations'

        except Exception as e:
            logger.error(f"Intelligent risk assessment failed: {e}")
            state.error = str(e)
            state.step = 'error'

    async def _memory_based_recommendations(self, state: MemoryEnhancedState, agent_logs: List[Dict[str, Any]]):
        """
        Generate recommendations based on memory patterns and MCP intelligence
        """
        try:
            recommendations = []

            # Base recommendations from compliance results
            if state.compliance_results:
                recommendations.extend(self._generate_compliance_recommendations(state.compliance_results))

            # Memory-based recommendations
            memory_recommendations = await self._generate_memory_recommendations(state.session_id)
            recommendations.extend(memory_recommendations)

            # MCP-powered recommendations if available
            if self.mcp_tools and state.compliance_results:
                mcp_response = await self.mcp_tools.get_compliance_recommendations(
                    state.compliance_results, state.session_id
                )
                if mcp_response.success:
                    recommendations.extend(mcp_response.data.get('recommendations', []))

            # Generate final insights
            state.memory_insights = await state.memory_agent.generate_insights(
                state.session_id,
                {
                    'dormant_results': state.dormant_results,
                    'compliance_results': state.compliance_results,
                    'confidence_score': state.confidence_score
                }
            )

            # Store recommendations in state
            state.result = {
                'recommendations': recommendations,
                'memory_insights': state.memory_insights,
                'session_summary': state.memory_agent.get_session_summary(state.session_id)
            }

            # Log completion
            state.memory_agent.log(
                state.session_id,
                'recommendations_generated',
                {
                    'recommendations_count': len(recommendations),
                    'memory_insights_generated': True,
                    'session_summary_created': True
                },
                tags=['recommendations', 'complete'],
                importance=1.0
            )

            agent_logs.append({
                'agent': 'memory_based_recommender',
                'timestamp': datetime.now().isoformat(),
                'status': 'success',
                'details': {
                    'recommendations_generated': len(recommendations),
                    'memory_insights': bool(state.memory_insights),
                    'mcp_enhanced': self.mcp_tools is not None
                }
            })

            state.step = 'complete'

        except Exception as e:
            logger.error(f"Memory-based recommendations failed: {e}")
            state.error = str(e)
            state.step = 'error'

    async def _generate_enhanced_insights(self, enhanced_state: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """
        Generate enhanced insights using memory and MCP
        """
        if not self.memory_agent:
            return enhanced_state

        try:
            # Generate memory insights
            insights = await self.memory_agent.generate_insights(session_id, enhanced_state)
            enhanced_state['memory_insights'] = insights

            # Get session summary
            enhanced_state['session_summary'] = self.memory_agent.get_session_summary(session_id)

            # Get knowledge stats
            enhanced_state['knowledge_stats'] = self.memory_agent.get_knowledge_stats()

            # Generate notifications based on insights
            notifications = self._generate_notifications_from_insights(insights, enhanced_state)
            enhanced_state['notifications'] = notifications

            return enhanced_state

        except Exception as e:
            logger.error(f"Failed to generate enhanced insights: {e}")
            return enhanced_state

    # ==================== Helper Methods ====================

    async def _get_historical_schemas(self, session_id: str) -> List[Dict[str, Any]]:
        """Get historical data schemas from memory"""
        if not self.memory_agent:
            return []

        historical_data = self.memory_agent.get(session_id, event_filter='data_processing_start')
        return [entry.get('data', {}) for entry in historical_data]

    def _validate_schema_with_history(self, data: pd.DataFrame, historical_schemas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate schema against historical patterns"""
        current_columns = set(data.columns)

        validation = {
            'valid': True,
            'missing_columns': [],
            'new_columns': [],
            'schema_drift': False
        }

        if historical_schemas:
            # Find most common historical columns
            all_historical_columns = set()
            for schema in historical_schemas:
                if 'columns' in schema:
                    all_historical_columns.update(schema['columns'])

            validation['missing_columns'] = list(all_historical_columns - current_columns)
            validation['new_columns'] = list(current_columns - all_historical_columns)
            validation['schema_drift'] = len(validation['missing_columns']) > 0 or len(validation['new_columns']) > 0

        return validation

    async def _assess_data_quality_with_memory(self, data: pd.DataFrame, session_id: str) -> Dict[str, Any]:
        """Assess data quality using memory patterns"""
        quality_assessment = {
            'overall_score': 0.8,
            'completeness': 1.0 - (data.isnull().sum().sum() / (len(data) * len(data.columns))),
            'consistency': 0.9,  # Would need more sophisticated analysis
            'accuracy': 0.85,   # Would need validation rules
            'enhancements': []
        }

        # Check for common data quality issues
        if quality_assessment['completeness'] < 0.95:
            quality_assessment['enhancements'].append('Fill missing values')

        if 'Account_ID' in data.columns and data['Account_ID'].duplicated().any():
            quality_assessment['enhancements'].append('Remove duplicate Account_IDs')
            quality_assessment['consistency'] *= 0.9

        # Calculate overall score
        quality_assessment['overall_score'] = (
            quality_assessment['completeness'] * 0.4 +
            quality_assessment['consistency'] * 0.3 +
            quality_assessment['accuracy'] * 0.3
        )

        return quality_assessment

    def _apply_data_enhancements(self, data: pd.DataFrame, quality_assessment: Dict[str, Any]) -> pd.DataFrame:
        """Apply data enhancements based on quality assessment"""
        enhanced_data = data.copy()

        for enhancement in quality_assessment.get('enhancements', []):
            if enhancement == 'Fill missing values':
                # Fill missing values with appropriate defaults
                for col in enhanced_data.columns:
                    if enhanced_data[col].dtype == 'object':
                        enhanced_data[col] = enhanced_data[col].fillna('')
                    else:
                        enhanced_data[col] = enhanced_data[col].fillna(0)

            elif enhancement == 'Remove duplicate Account_IDs' and 'Account_ID' in enhanced_data.columns:
                enhanced_data = enhanced_data.drop_duplicates(subset=['Account_ID'])

        return enhanced_data

    async def _get_historical_dormancy_patterns(self, session_id: str) -> Optional[pd.DataFrame]:
        """Get historical dormancy patterns from memory"""
        if not self.memory_agent:
            return None

        # This would ideally query historical dormancy data
        # For now, return None to use default behavior
        return None

    def _extract_dormant_accounts(self, dormant_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract dormant accounts for MCP analysis"""
        dormant_accounts = []

        for check_key in ['sdb_dormant', 'investment_dormant', 'fixed_deposit_dormant',
                          'demand_deposit_dormant', 'unclaimed_instruments']:
            if check_key in dormant_results and 'df' in dormant_results[check_key]:
                df = dormant_results[check_key]['df']
                if not df.empty:
                    accounts = df.to_dict('records')
                    for account in accounts:
                        account['dormancy_type'] = check_key
                    dormant_accounts.extend(accounts)

        return dormant_accounts

    async def _enhance_compliance_with_mcp(self, compliance_results: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Enhance compliance results with MCP validation"""
        enhanced_results = {}

        try:
            # Validate each compliance check with MCP
            for check_name, check_result in compliance_results.items():
                if isinstance(check_result, dict) and 'df' in check_result:
                    df = check_result['df']
                    if not df.empty:
                        # Convert to records for MCP validation
                        accounts = df.to_dict('records')

                        # Batch validate with MCP
                        mcp_responses = await self.mcp_tools.batch_compliance_check(
                            accounts, session_id, batch_size=50
                        )

                        # Process MCP responses
                        mcp_validation = self._process_mcp_batch_responses(mcp_responses)
                        enhanced_results[f"{check_name}_mcp_validation"] = mcp_validation

        except Exception as e:
            logger.error(f"MCP compliance enhancement failed: {e}")

        return enhanced_results

    def _process_mcp_batch_responses(self, mcp_responses: List) -> Dict[str, Any]:
        """Process batch MCP responses"""
        validation_summary = {
            'total_validated': 0,
            'validation_success_rate': 0,
            'critical_issues': [],
            'recommendations': []
        }

        successful_validations = 0
        total_validations = 0

        for response in mcp_responses:
            if hasattr(response, 'success') and response.success:
                successful_validations += 1
                # Process response data
                if hasattr(response, 'data') and 'critical_issues' in response.data:
                    validation_summary['critical_issues'].extend(response.data['critical_issues'])
                if hasattr(response, 'data') and 'recommendations' in response.data:
                    validation_summary['recommendations'].extend(response.data['recommendations'])
            total_validations += 1

        if total_validations > 0:
            validation_summary['validation_success_rate'] = successful_validations / total_validations

        validation_summary['total_validated'] = total_validations

        return validation_summary

    def _calculate_base_confidence(self, state: MemoryEnhancedState) -> float:
        """Calculate base confidence score"""
        confidence = 0.8

        # Reduce confidence if there are errors
        if state.error:
            confidence -= 0.3

        # Increase confidence based on data quality
        if state.enhanced_data is not None:
            confidence += 0.1

        # Adjust based on dormant and compliance results
        if state.dormant_results:
            confidence += 0.05
        if state.compliance_results:
            confidence += 0.05

        return min(max(confidence, 0.0), 1.0)

    async def _calculate_memory_confidence(self, session_id: str) -> float:
        """Calculate confidence based on memory patterns"""
        if not self.memory_agent:
            return 0.8

        try:
            session_summary = self.memory_agent.get_session_summary(session_id)

            # Higher confidence for sessions with more entries and higher importance
            entry_count = session_summary.get('total_entries', 0)
            avg_importance = session_summary.get('average_importance', 0.5)

            # Calculate confidence based on session quality
            confidence = 0.6 + (min(entry_count, 20) / 20) * 0.2 + avg_importance * 0.2

            return min(max(confidence, 0.0), 1.0)
        except Exception as e:
            logger.error(f"Memory confidence calculation failed: {e}")
            return 0.8

    async def _calculate_mcp_confidence(self, state: MemoryEnhancedState) -> float:
        """Calculate confidence based on MCP validation"""
        if not self.mcp_tools:
            return 0.8

        try:
            # Get MCP performance stats
            mcp_stats = self.mcp_tools.get_performance_stats()
            success_rate = mcp_stats.get('success_rate', 80) / 100

            # Base confidence on MCP performance
            confidence = 0.6 + success_rate * 0.4

            return min(max(confidence, 0.0), 1.0)
        except Exception as e:
            logger.error(f"MCP confidence calculation failed: {e}")
            return 0.8

    async def _generate_risk_indicators(self, state: MemoryEnhancedState) -> List[str]:
        """Generate risk indicators"""
        indicators = []

        # Compliance-based indicators
        if state.compliance_results:
            cr = state.compliance_results
            if cr.get('transfer_candidates_cb', {}).get('count', 0) > 0:
                indicators.append('CBUAE transfer candidates identified')
            if cr.get('incomplete_contact', {}).get('count', 0) > 0:
                indicators.append('Incomplete contact attempts detected')

        # Dormancy-based indicators
        if state.dormant_results:
            dr = state.dormant_results
            if dr['summary_kpis'].get('count_high_value_dormant', 0) > 0:
                indicators.append('High-value dormant accounts present')

        # Confidence-based indicators
        if state.confidence_score < 0.7:
            indicators.append('Low confidence in analysis results')

        return indicators

    def _generate_compliance_recommendations(self, compliance_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations from compliance results"""
        recommendations = []

        if compliance_results.get('transfer_candidates_cb', {}).get('count', 0) > 0:
            recommendations.append('URGENT: Initiate CBUAE transfer process for eligible accounts')

        if compliance_results.get('incomplete_contact', {}).get('count', 0) > 0:
            recommendations.append('Complete outstanding contact attempts for dormant accounts')

        if compliance_results.get('flag_candidates', {}).get('count', 0) > 0:
            recommendations.append('Flag identified accounts as dormant in the system')

        return recommendations

    async def _generate_memory_recommendations(self, session_id: str) -> List[str]:
        """Generate recommendations based on memory patterns"""
        if not self.memory_agent:
            return []

        recommendations = []

        try:
            # Analyze session patterns
            session_summary = self.memory_agent.get_session_summary(session_id)

            # Check for recurring issues
            event_distribution = session_summary.get('event_distribution', {})
            if event_distribution.get('error', 0) > 0:
                recommendations.append('Review and address recurring processing errors')

            # Check session efficiency
            if session_summary.get('average_importance', 0.5) < 0.6:
                recommendations.append('Optimize data processing workflow for better results')

        except Exception as e:
            logger.error(f"Memory recommendation generation failed: {e}")

        return recommendations

    def _generate_notifications_from_insights(self, insights: Dict[str, Any], state: Dict[str, Any]) -> List[str]:
        """Generate notifications from insights"""
        notifications = []

        # MCP-enhanced notifications
        if insights.get('mcp_enhanced_report'):
            mcp_report = insights['mcp_enhanced_report']
            if isinstance(mcp_report, dict) and mcp_report.get('critical_alerts'):
                notifications.extend(mcp_report['critical_alerts'])

        # Memory pattern notifications
        session_patterns = insights.get('session_patterns', {})
        if session_patterns.get('error_rate', 0) > 0.1:
            notifications.append('⚠️ High error rate detected in current session')

        # Standard compliance notifications
        compliance_results = state.get('compliance_results', {})
        if compliance_results.get('transfer_candidates_cb', {}).get('count', 0) > 0:
            count = compliance_results['transfer_candidates_cb']['count']
            notifications.append(f'🚨 URGENT: {count} accounts require CBUAE transfer')

        return notifications

    def _convert_to_ui_format(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Convert state to UI-compatible format"""
        return {
            'session_id': state_dict.get('session_id'),
            'data': state_dict.get('data'),
            'enhanced_data': state_dict.get('enhanced_data'),
            'dormant_results': state_dict.get('dormant_results'),
            'compliance_results': state_dict.get('compliance_results'),
            'non_compliant_data': state_dict.get('non_compliant_data'),
            'current_step': 'completed',
            'messages': [],
            'error': state_dict.get('error'),
            'confidence_score': state_dict.get('confidence_score', 0.8),
            'timestamp': datetime.now(),
            'agent_logs': [],
            'notifications': state_dict.get('notifications', []),
            'final_result': state_dict.get('result', {}).get('recommendations', ['Analysis completed successfully']),
            'recommendations': state_dict.get('result', {}).get('recommendations', []),
            'mcp_enabled': self.mcp_tools is not None,
            'mcp_results': state_dict.get('mcp_insights'),
            'memory_insights': state_dict.get('memory_insights'),
            'session_summary': state_dict.get('session_summary'),
            'knowledge_stats': state_dict.get('knowledge_stats')
        }

    def _enhance_legacy_state(self, legacy_state, legacy_memory, session_id: str) -> Dict[str, Any]:
        """Enhance legacy state with memory integration"""
        enhanced_state = {
            'session_id': session_id,
            'data': legacy_state.data,
            'enhanced_data': getattr(legacy_state, 'enhanced_data', None),
            'dormant_results': getattr(legacy_state, 'dormant_results', None),
            'compliance_results': getattr(legacy_state, 'compliance_results', None),
            'non_compliant_data': getattr(legacy_state, 'non_compliant_data', None),
            'error': getattr(legacy_state, 'error', None),
            'confidence_score': self._calculate_legacy_confidence(legacy_state),
            'result': getattr(legacy_state, 'result', None)
        }

        return enhanced_state

    def _extract_enhanced_agent_logs(self, legacy_state, legacy_memory) -> List[Dict[str, Any]]:
        """Extract enhanced agent logs from legacy execution"""
        logs = []

        if legacy_memory and hasattr(legacy_state, 'session_id'):
            memory_logs = legacy_memory.get(legacy_state.session_id)
            if memory_logs:
                for i, log_entry in enumerate(memory_logs):
                    logs.append({
                        'agent': f"legacy_{log_entry.get('event', f'step_{i}')}",
                        'timestamp': datetime.now().isoformat(),
                        'status': 'success',
                        'details': log_entry.get('data', {})
                    })

        return logs

    def _calculate_legacy_confidence(self, legacy_state) -> float:
        """Calculate confidence for legacy state"""
        confidence = 0.7  # Base confidence for legacy mode

        if getattr(legacy_state, 'error', None):
            confidence -= 0.2
        if getattr(legacy_state, 'dormant_results', None):
            confidence += 0.1
        if getattr(legacy_state, 'compliance_results', None):
            confidence += 0.1

        return min(max(confidence, 0.0), 1.0)

    def _create_error_state(self, error_message: str, session_id: str) -> Dict[str, Any]:
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
            'error': error_message,
            'confidence_score': 0.0,
            'timestamp': datetime.now(),
            'agent_logs': [{
                'agent': 'error_handler',
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'details': {'error': error_message}
            }],
            'notifications': [f'❌ Analysis failed: {error_message}'],
            'final_result': f'Analysis failed: {error_message}',
            'recommendations': ['Check system configuration', 'Review input data'],
            'mcp_enabled': False,
            'mcp_results': None
        }

    def _update_orchestrator_stats(self, account_count: int, processing_time: float):
        """Update orchestrator statistics"""
        self.orchestrator_stats['sessions_processed'] += 1
        self.orchestrator_stats['total_accounts_analyzed'] += account_count

        # Update average processing time
        sessions = self.orchestrator_stats['sessions_processed']
        current_avg = self.orchestrator_stats['average_processing_time']
        self.orchestrator_stats['average_processing_time'] = (
            (current_avg * (sessions - 1)) + processing_time
        ) / sessions

    def get_orchestrator_stats(self) -> Dict[str, Any]:
        """Get orchestrator performance statistics"""
        stats = self.orchestrator_stats.copy()

        if self.memory_agent:
            stats['memory_stats'] = self.memory_agent.get_knowledge_stats()

        if self.mcp_tools:
            stats['mcp_stats'] = self.mcp_tools.get_performance_stats()

        return stats

    async def cleanup(self):
        """Cleanup resources"""
        if self.memory_agent:
            await self.memory_agent.cleanup_expired_entries()
            await self.memory_agent.close()

        if self.mcp_tools:
            await self.mcp_tools.close()

        logger.info("Memory-enhanced orchestrator cleanup completed")


# Factory function for backward compatibility
async def create_memory_enhanced_orchestrator(use_enhanced: bool = True) -> MemoryEnhancedComplianceOrchestrator:
    """Create and initialize memory-enhanced orchestrator"""
    orchestrator = MemoryEnhancedComplianceOrchestrator(use_enhanced=use_enhanced)
    return orchestrator


# Backward compatibility function
async def run_memory_enhanced_flow(data: pd.DataFrame) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run memory-enhanced flow - async version"""
    orchestrator = await create_memory_enhanced_orchestrator()
    return await orchestrator.run_flow_enhanced(data)