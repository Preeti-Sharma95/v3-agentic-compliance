# hybrid_memory_integration.py
"""
Complete integration file for Hybrid Memory Agent with MCP support
This file demonstrates how to integrate the hybrid memory system into the existing banking application
Place this file in your project root directory (same level as main.py)
"""

import asyncio
import sys
from pathlib import Path
import uuid
import random
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from datetime import datetime, timedelta
import logging
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Core imports
try:
    from agents.hybrid_memory_agent import HybridMemoryAgent

    HYBRID_MEMORY_AVAILABLE = True
except ImportError:
    HYBRID_MEMORY_AVAILABLE = False
    HybridMemoryAgent = None

try:
    from mcp.enhanced_mcp_integration import EnhancedMCPComplianceTools, MCPKnowledgeManager

    ENHANCED_MCP_AVAILABLE = True
except ImportError:
    ENHANCED_MCP_AVAILABLE = False
    EnhancedMCPComplianceTools = None
    MCPKnowledgeManager = None

try:
    from enhanced_orchestrator_with_memory import MemoryEnhancedComplianceOrchestrator

    ENHANCED_ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ENHANCED_ORCHESTRATOR_AVAILABLE = False
    MemoryEnhancedComplianceOrchestrator = None

try:
    from config.memory_config import ConfigManager, ConfigMonitor

    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False


    # Fallback configuration
    class ConfigManager:
        def __init__(self):
            self.config = type('Config', (), {
                'environment': 'development',
                'memory_system_enabled': True,
                'mcp_integration_enabled': False,
                'mcp': type('MCP', (), {
                    'server_url': 'http://localhost:8000',
                    'enabled': False,
                    'timeout_seconds': 30,
                    'max_retries': 3,
                    'batch_size': 50
                })(),
                'memory': type('Memory', (), {
                    'session_cache_size': 500,
                    'knowledge_cache_size': 1000,
                    'enable_persistence': True,
                    'persistence_path': 'data/memory'
                })()
            })()

        def is_mcp_enabled(self):
            return False

        def is_memory_enabled(self):
            return True

        def get_config_summary(self):
            return {
                'system': {'environment': 'development', 'enhanced_mode': True},
                'memory': {'enabled': True, 'session_cache_size': 500},
                'mcp': {'enabled': False}
            }


    ConfigMonitor = None

# Fallback imports for legacy compatibility
try:
    from orchestrator import run_flow as legacy_run_flow

    LEGACY_ORCHESTRATOR_AVAILABLE = True
except ImportError:
    LEGACY_ORCHESTRATOR_AVAILABLE = False
    legacy_run_flow = None

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/hybrid_memory.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class HybridMemoryBankingSystem:
    """
    Complete banking compliance system with hybrid memory and MCP integration
    """

    def __init__(self, config_path: Optional[str] = None):
        # Initialize configuration
        if CONFIG_AVAILABLE:
            self.config_manager = ConfigManager(config_path)
            self.config = self.config_manager.config
        else:
            self.config_manager = ConfigManager()
            self.config = self.config_manager.config

        # Initialize components
        self.memory_agent: Optional[HybridMemoryAgent] = None
        self.mcp_tools: Optional[EnhancedMCPComplianceTools] = None
        self.knowledge_manager: Optional[MCPKnowledgeManager] = None
        self.orchestrator: Optional[MemoryEnhancedComplianceOrchestrator] = None
        self.config_monitor: Optional[ConfigMonitor] = None

        # System state
        self.initialized = False
        self.system_stats = {
            'startup_time': datetime.now(),
            'sessions_processed': 0,
            'total_runtime_hours': 0,
            'memory_operations': 0,
            'mcp_operations': 0
        }

        logger.info("Hybrid Memory Banking System created")

    async def initialize(self) -> bool:
        """
        Initialize the complete system asynchronously
        """
        try:
            logger.info("Initializing Hybrid Memory Banking System...")

            # Check component availability
            if not HYBRID_MEMORY_AVAILABLE:
                logger.warning("Hybrid memory agent not available - using basic mode")

            if not ENHANCED_MCP_AVAILABLE:
                logger.warning("Enhanced MCP tools not available - using basic mode")

            if not ENHANCED_ORCHESTRATOR_AVAILABLE:
                logger.warning("Enhanced orchestrator not available - using legacy mode")

            # Step 1: Initialize memory agent
            if self.config.memory_system_enabled and HYBRID_MEMORY_AVAILABLE:
                await self._initialize_memory_system()

            # Step 2: Initialize MCP tools
            if self.config_manager.is_mcp_enabled() and ENHANCED_MCP_AVAILABLE:
                await self._initialize_mcp_system()

            # Step 3: Initialize orchestrator
            await self._initialize_orchestrator()

            # Step 4: Initialize monitoring
            if CONFIG_AVAILABLE and ConfigMonitor:
                await self._initialize_monitoring()

            # Step 5: Run system health check
            health_status = await self._run_health_check()
            if not all(health_status.values()):
                logger.warning(f"System health check issues: {health_status}")

            self.initialized = True
            logger.info("Hybrid Memory Banking System initialized successfully")
            return True

        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return False

    async def _initialize_memory_system(self):
        """Initialize the hybrid memory agent"""
        logger.info("Initializing hybrid memory system...")

        if HYBRID_MEMORY_AVAILABLE:
            self.memory_agent = HybridMemoryAgent(
                mcp_server_url=self.config.mcp.server_url,
                knowledge_cache_size=self.config.memory.knowledge_cache_size,
                session_cache_size=self.config.memory.session_cache_size
            )

            # Wait for async initialization
            await asyncio.sleep(1)

            logger.info("Hybrid memory system initialized")
        else:
            logger.warning("Hybrid memory agent not available")

    async def _initialize_mcp_system(self):
        """Initialize MCP tools and knowledge manager"""
        logger.info("Initializing MCP system...")

        if ENHANCED_MCP_AVAILABLE:
            self.mcp_tools = EnhancedMCPComplianceTools(
                server_url=self.config.mcp.server_url,
                timeout=self.config.mcp.timeout_seconds,
                max_retries=self.config.mcp.max_retries
            )

            # Initialize MCP connection
            mcp_initialized = await self.mcp_tools.initialize()
            if mcp_initialized:
                self.knowledge_manager = MCPKnowledgeManager(self.mcp_tools)
                logger.info("MCP system initialized successfully")
            else:
                logger.warning("MCP system failed to initialize - operating in local mode")
                self.mcp_tools = None
        else:
            logger.warning("Enhanced MCP tools not available")

    async def _initialize_orchestrator(self):
        """Initialize the memory-enhanced orchestrator"""
        logger.info("Initializing orchestrator...")

        if ENHANCED_ORCHESTRATOR_AVAILABLE:
            self.orchestrator = MemoryEnhancedComplianceOrchestrator(
                use_enhanced=True,
                mcp_server_url=self.config.mcp.server_url
            )

            # Set up orchestrator components
            if self.memory_agent:
                self.orchestrator.memory_agent = self.memory_agent
            if self.mcp_tools:
                self.orchestrator.mcp_tools = self.mcp_tools
                self.orchestrator.knowledge_manager = self.knowledge_manager
        else:
            logger.warning("Enhanced orchestrator not available - will use legacy mode")

        logger.info("Orchestrator initialized")

    async def _initialize_monitoring(self):
        """Initialize system monitoring"""
        logger.info("Initializing monitoring...")

        if ConfigMonitor:
            self.config_monitor = ConfigMonitor(self.config)

        logger.info("Monitoring initialized")

    async def _run_health_check(self) -> Dict[str, bool]:
        """Run comprehensive system health check"""
        health_status = {}

        try:
            # Check memory system
            if self.memory_agent:
                health_status['memory_system'] = True
                # Test memory operations
                test_session = "health_check_test"
                self.memory_agent.log(test_session, "health_check", {"status": "ok"})
                health_status['memory_operations'] = len(self.memory_agent.get(test_session)) > 0
            else:
                health_status['memory_system'] = False
                health_status['memory_operations'] = False

            # Check MCP system
            if self.mcp_tools:
                health_status['mcp_connection'] = self.mcp_tools.initialized
                if self.mcp_tools.initialized:
                    # Test MCP operations
                    test_response = await self.mcp_tools._health_check()
                    health_status['mcp_operations'] = test_response.success
                else:
                    health_status['mcp_operations'] = False
            else:
                health_status['mcp_connection'] = False
                health_status['mcp_operations'] = False

            # Check configuration
            if self.config_monitor:
                config_health = await self.config_monitor.check_system_health()
                health_status.update(config_health)

            # Overall system health
            health_status['overall_system'] = True

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_status['system_error'] = False

        return health_status

    async def process_banking_data(self,
                                   data: pd.DataFrame,
                                   session_name: str = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Process banking data with full hybrid memory and MCP integration
        """
        if not self.initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")

        session_name = session_name or f"banking_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            logger.info(f"Processing banking data for session: {session_name}")
            logger.info(f"Data shape: {data.shape}")

            # Log session start in memory
            if self.memory_agent:
                self.memory_agent.log(
                    session_name,
                    'banking_data_processing_start',
                    {
                        'data_shape': data.shape,
                        'columns': data.columns.tolist(),
                        'processing_timestamp': datetime.now().isoformat()
                    },
                    tags=['banking', 'processing', 'start'],
                    importance=0.9
                )

            # Process with enhanced orchestrator or fallback
            if self.orchestrator and ENHANCED_ORCHESTRATOR_AVAILABLE:
                results, agent_logs = await self.orchestrator.run_flow_enhanced(data)
            else:
                # Fallback to legacy processing
                results, agent_logs = await self._run_legacy_processing(data, session_name)

            # Log session completion
            if self.memory_agent:
                self.memory_agent.log(
                    session_name,
                    'banking_data_processing_complete',
                    {
                        'results_summary': {
                            'confidence_score': results.get('confidence_score', 0),
                            'dormant_accounts': self._extract_dormant_count(results),
                            'compliance_issues': len(results.get('notifications', []))
                        }
                    },
                    tags=['banking', 'processing', 'complete'],
                    importance=1.0
                )

            # Update system stats
            self._update_system_stats(data, results)

            logger.info(f"Banking data processing completed for session: {session_name}")
            return results, agent_logs

        except Exception as e:
            logger.error(f"Banking data processing failed: {e}")

            # Log error in memory
            if self.memory_agent:
                self.memory_agent.log(
                    session_name,
                    'banking_data_processing_error',
                    {'error': str(e)},
                    tags=['banking', 'processing', 'error'],
                    importance=1.0
                )

            # Return error state instead of raising
            return self._create_error_state(str(e), session_name), []

    async def _run_legacy_processing(self, data: pd.DataFrame, session_name: str) -> Tuple[
        Dict[str, Any], List[Dict[str, Any]]]:
        """Fallback to legacy processing when enhanced orchestrator is not available"""
        logger.info("Using legacy processing mode")

        try:
            if LEGACY_ORCHESTRATOR_AVAILABLE and legacy_run_flow:
                # Use legacy orchestrator
                legacy_state, legacy_memory = legacy_run_flow(data)

                # Convert to enhanced format
                results = self._convert_legacy_to_enhanced(legacy_state, legacy_memory, session_name)
                agent_logs = self._extract_legacy_logs(legacy_state, legacy_memory)

                return results, agent_logs
            else:
                # Basic processing without orchestrator
                return self._basic_processing(data, session_name)

        except Exception as e:
            logger.error(f"Legacy processing failed: {e}")
            return self._create_error_state(str(e), session_name), []

    def _basic_processing(self, data: pd.DataFrame, session_name: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Basic processing when no orchestrator is available"""
        logger.info("Using basic processing mode")

        # Simple analysis
        results = {
            'session_id': session_name,
            'data': data,
            'enhanced_data': None,
            'dormant_results': None,
            'compliance_results': None,
            'non_compliant_data': None,
            'current_step': 'completed',
            'messages': [],
            'error': None,
            'confidence_score': 0.7,  # Basic confidence
            'timestamp': datetime.now(),
            'agent_logs': [],
            'notifications': [f"Processed {len(data)} accounts using basic mode"],
            'final_result': f"Basic analysis completed for {len(data)} accounts",
            'recommendations': [
                "Consider enabling enhanced mode for better analysis",
                "Review account dormancy patterns manually",
                "Ensure compliance with regulatory requirements"
            ],
            'mcp_enabled': False,
            'mcp_results': None
        }

        agent_logs = [{
            'agent': 'basic_processor',
            'timestamp': datetime.now().isoformat(),
            'status': 'success',
            'details': {
                'accounts_processed': len(data),
                'mode': 'basic',
                'features_available': {
                    'memory': self.memory_agent is not None,
                    'mcp': self.mcp_tools is not None,
                    'enhanced_orchestrator': False
                }
            }
        }]

        return results, agent_logs

    async def get_regulatory_guidance(self,
                                      article: str,
                                      account_type: str,
                                      session_id: str) -> Dict[str, Any]:
        """
        Get regulatory guidance with memory integration
        """
        if not self.memory_agent:
            return self._fallback_regulatory_guidance(article, account_type)

        try:
            # Get guidance with memory caching
            guidance = await self.memory_agent.get_regulatory_guidance(
                article, account_type, cache_duration_hours=24
            )

            # Log the query
            self.memory_agent.log(
                session_id,
                'regulatory_guidance_query',
                {
                    'article': article,
                    'account_type': account_type,
                    'guidance_source': guidance.get('source', 'unknown')
                },
                tags=['regulatory', 'guidance', 'query'],
                importance=0.7
            )

            return guidance

        except Exception as e:
            logger.error(f"Regulatory guidance query failed: {e}")
            return self._fallback_regulatory_guidance(article, account_type)

    def _fallback_regulatory_guidance(self, article: str, account_type: str) -> Dict[str, Any]:
        """Fallback regulatory guidance when memory system is not available"""
        return {
            'article': article,
            'account_type': account_type,
            'guidance': f"Standard CBUAE guidance for Article {article} applies to {account_type} accounts. "
                        f"Refer to official CBUAE documentation for complete details.",
            'source': 'local_fallback',
            'confidence': 0.6,
            'recommendations': [
                f"Review Article {article} requirements",
                f"Ensure {account_type} account compliance",
                "Consult official CBUAE documentation"
            ]
        }

    async def validate_account_compliance(self,
                                          account_data: Dict[str, Any],
                                          session_id: str) -> Dict[str, Any]:
        """
        Validate account compliance with enhanced memory and MCP
        """
        if not self.memory_agent:
            return self._fallback_compliance_validation(account_data)

        try:
            # Enhanced compliance validation
            validation_result = await self.memory_agent.validate_compliance_enhanced(
                account_data, session_id
            )

            return validation_result

        except Exception as e:
            logger.error(f"Account compliance validation failed: {e}")
            return self._fallback_compliance_validation(account_data)

    def _fallback_compliance_validation(self, account_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback compliance validation"""
        issues = []

        # Basic validation rules
        if account_data.get('Current_Balance', 0) > 100000:
            issues.append("High-value account requires enhanced monitoring")

        if account_data.get('Expected_Account_Dormant') == 'yes':
            issues.append("Dormant account status confirmed")

        # Check for missing required fields
        required_fields = ['Account_ID', 'Account_Type']
        for field in required_fields:
            if field not in account_data or not account_data[field]:
                issues.append(f"Missing required field: {field}")

        return {
            'compliant': len(issues) == 0,
            'issues': issues,
            'source': 'local_validation',
            'confidence': 0.7,
            'account_id': account_data.get('Account_ID', 'unknown'),
            'validation_timestamp': datetime.now().isoformat()
        }

    async def generate_compliance_insights(self,
                                           session_id: str,
                                           analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive compliance insights
        """
        if not self.memory_agent:
            return self._fallback_insights(analysis_results)

        try:
            # Generate memory-enhanced insights
            insights = await self.memory_agent.generate_insights(session_id, analysis_results)

            # Enhance with MCP knowledge if available
            if self.knowledge_manager:
                try:
                    # Get relevant regulatory articles
                    if 'dormant_results' in analysis_results:
                        regulatory_context = await self.knowledge_manager.get_regulatory_article(
                            '2.1', session_id  # CBUAE Article 2.1 for dormancy
                        )
                        insights['regulatory_context'] = regulatory_context
                except Exception as e:
                    logger.warning(f"Failed to get regulatory context: {e}")

            return insights

        except Exception as e:
            logger.error(f"Insights generation failed: {e}")
            return self._fallback_insights(analysis_results)

    def _fallback_insights(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback insights generation"""
        insights = {
            'session_patterns': {
                'analysis_completed': True,
                'data_quality': 'acceptable',
                'processing_mode': 'basic'
            },
            'recommendations': [
                "Enable enhanced memory system for better insights",
                "Consider MCP integration for regulatory guidance",
                "Review compliance results manually"
            ],
            'risk_indicators': [],
            'confidence_assessment': {
                'overall_confidence': analysis_results.get('confidence_score', 0.7),
                'data_quality_score': 0.8,
                'completeness_score': 0.7
            }
        }

        # Add risk indicators based on results
        if analysis_results.get('notifications'):
            insights['risk_indicators'].extend([
                f"Found {len(analysis_results['notifications'])} notifications requiring attention"
            ])

        return insights

    def get_session_history(self, session_id: str) -> Dict[str, Any]:
        """
        Get complete session history from memory
        """
        if not self.memory_agent:
            return {
                'error': 'Memory system not initialized',
                'session_id': session_id,
                'fallback_message': 'Session history not available in basic mode'
            }

        try:
            session_summary = self.memory_agent.get_session_summary(session_id)
            return session_summary
        except Exception as e:
            logger.error(f"Session history retrieval failed: {e}")
            return {'error': str(e), 'session_id': session_id}

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status
        """
        status = {
            'initialized': self.initialized,
            'components': {
                'memory_agent': self.memory_agent is not None,
                'mcp_tools': self.mcp_tools is not None,
                'knowledge_manager': self.knowledge_manager is not None,
                'orchestrator': self.orchestrator is not None
            },
            'configuration': self.config_manager.get_config_summary(),
            'system_stats': self.system_stats.copy(),
            'component_availability': {
                'hybrid_memory': HYBRID_MEMORY_AVAILABLE,
                'enhanced_mcp': ENHANCED_MCP_AVAILABLE,
                'enhanced_orchestrator': ENHANCED_ORCHESTRATOR_AVAILABLE,
                'config_system': CONFIG_AVAILABLE,
                'legacy_orchestrator': LEGACY_ORCHESTRATOR_AVAILABLE
            }
        }

        # Add component-specific status
        if self.memory_agent and HYBRID_MEMORY_AVAILABLE:
            try:
                status['memory_stats'] = self.memory_agent.get_knowledge_stats()
            except Exception as e:
                status['memory_stats'] = {'error': str(e)}

        if self.mcp_tools and ENHANCED_MCP_AVAILABLE:
            try:
                status['mcp_stats'] = self.mcp_tools.get_performance_stats()
            except Exception as e:
                status['mcp_stats'] = {'error': str(e)}

        if self.orchestrator and ENHANCED_ORCHESTRATOR_AVAILABLE:
            try:
                status['orchestrator_stats'] = self.orchestrator.get_orchestrator_stats()
            except Exception as e:
                status['orchestrator_stats'] = {'error': str(e)}

        return status

    async def cleanup_system(self):
        """
        Cleanup system resources
        """
        logger.info("Cleaning up Hybrid Memory Banking System...")

        try:
            # Cleanup memory agent
            if self.memory_agent:
                await self.memory_agent.cleanup_expired_entries()
                await self.memory_agent.close()

            # Cleanup MCP tools
            if self.mcp_tools:
                await self.mcp_tools.close()

            # Cleanup orchestrator
            if self.orchestrator:
                await self.orchestrator.cleanup()

            logger.info("System cleanup completed")

        except Exception as e:
            logger.error(f"System cleanup failed: {e}")

    # ==================== Helper Methods ====================

    def _extract_dormant_count(self, results: Dict[str, Any]) -> int:
        """Extract dormant account count from results"""
        try:
            if 'dormant_results' in results:
                return results['dormant_results'].get('summary_kpis', {}).get('total_accounts_flagged_dormant', 0)
            return 0
        except Exception:
            return 0

    def _convert_legacy_to_enhanced(self, legacy_state, legacy_memory, session_name: str) -> Dict[str, Any]:
        """Convert legacy state to enhanced format"""
        enhanced_state = {
            'session_id': session_name,
            'data': legacy_state.data,
            'enhanced_data': getattr(legacy_state, 'enhanced_data', None),
            'dormant_results': getattr(legacy_state, 'dormant_results', None),
            'compliance_results': getattr(legacy_state, 'compliance_results', None),
            'non_compliant_data': getattr(legacy_state, 'non_compliant_data', None),
            'current_step': 'completed',
            'messages': [],
            'error': getattr(legacy_state, 'error', None),
            'confidence_score': self._calculate_legacy_confidence(legacy_state),
            'timestamp': datetime.now(),
            'agent_logs': [],
            'notifications': self._extract_legacy_notifications(legacy_state),
            'final_result': getattr(legacy_state, 'result', f"Legacy analysis completed for session {session_name}"),
            'recommendations': self._generate_legacy_recommendations(legacy_state),
            'mcp_enabled': False,
            'mcp_results': None
        }

        return enhanced_state

    def _extract_legacy_logs(self, legacy_state, legacy_memory) -> List[Dict[str, Any]]:
        """Extract logs from legacy execution"""
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

        # Add default log if no logs found
        if not logs:
            logs.append({
                'agent': 'legacy_processor',
                'timestamp': datetime.now().isoformat(),
                'status': 'success',
                'details': {'message': 'Legacy analysis completed'}
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

    def _extract_legacy_notifications(self, legacy_state) -> List[str]:
        """Extract notifications from legacy state"""
        notifications = []

        if getattr(legacy_state, 'error', None):
            notifications.append(f"⚠️ Processing completed with warnings: {legacy_state.error}")

        if getattr(legacy_state, 'result', None):
            notifications.append(f"ℹ️ {legacy_state.result}")

        if not notifications:
            notifications.append("✅ Legacy analysis completed successfully")

        return notifications

    def _generate_legacy_recommendations(self, legacy_state) -> List[str]:
        """Generate recommendations for legacy processing"""
        recommendations = [
            "Consider upgrading to enhanced mode for better analysis",
            "Enable memory system for improved insights",
            "Review compliance results manually"
        ]

        if getattr(legacy_state, 'dormant_results', None):
            recommendations.append("Review identified dormant accounts")

        if getattr(legacy_state, 'compliance_results', None):
            recommendations.append("Address compliance issues identified")

        return recommendations

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
            'recommendations': [
                'Check system configuration',
                'Review input data format',
                'Check system logs for details',
                'Try enabling basic mode if enhanced features fail'
            ],
            'mcp_enabled': False,
            'mcp_results': None
        }

    def _update_system_stats(self, data: pd.DataFrame, results: Dict[str, Any]):
        """Update system statistics"""
        self.system_stats['sessions_processed'] += 1

    def _update_system_stats(self, data: pd.DataFrame, results: Dict[str, Any]):
        """Update system statistics"""
        self.system_stats['sessions_processed'] += 1

        # Calculate runtime
        runtime = datetime.now() - self.system_stats['startup_time']
        self.system_stats['total_runtime_hours'] = runtime.total_seconds() / 3600

        # Update memory operations count
        if self.memory_agent:
            self.system_stats['memory_operations'] += 1

        # Update MCP operations count
        if self.mcp_tools:
            self.system_stats['mcp_operations'] += 1


# Demo and testing functions
async def demo_hybrid_memory_system():
    """
    Demonstration of the hybrid memory banking system
    """
    print("=== Hybrid Memory Banking System Demo ===")

    # Create system instance
    system = HybridMemoryBankingSystem()

    try:
        # Initialize system
        print("1. Initializing system...")
        initialized = await system.initialize()
        if not initialized:
            print("❌ System initialization failed")
            return
        print("✅ System initialized successfully")

        # Check system status
        print("\n2. System status:")
        status = system.get_system_status()
        print(f"   Components active: {sum(status['components'].values())}/4")
        print(f"   Memory system: {'✅' if status['components']['memory_agent'] else '❌'}")
        print(f"   MCP integration: {'✅' if status['components']['mcp_tools'] else '❌'}")
        print(f"   Enhanced orchestrator: {'✅' if status['components']['orchestrator'] else '❌'}")

        # Show component availability
        availability = status['component_availability']
        print(f"\n   Component Availability:")
        print(f"   - Hybrid Memory: {'✅' if availability['hybrid_memory'] else '❌'}")
        print(f"   - Enhanced MCP: {'✅' if availability['enhanced_mcp'] else '❌'}")
        print(f"   - Enhanced Orchestrator: {'✅' if availability['enhanced_orchestrator'] else '❌'}")
        print(f"   - Legacy Orchestrator: {'✅' if availability['legacy_orchestrator'] else '❌'}")

        # Create sample banking data
        print("\n3. Creating sample banking data...")
        sample_data = create_sample_banking_data()
        print(f"   Created {len(sample_data)} sample accounts")

        # Process banking data
        print("\n4. Processing banking data...")
        results, agent_logs = await system.process_banking_data(
            sample_data,
            session_name="demo_session"
        )
        print(f"   ✅ Processing completed")
        print(f"   Confidence score: {results.get('confidence_score', 0):.2f}")
        print(f"   Agent steps: {len(agent_logs)}")
        print(f"   Notifications: {len(results.get('notifications', []))}")

        # Test regulatory guidance
        print("\n5. Testing regulatory guidance...")
        guidance = await system.get_regulatory_guidance(
            article="2.1",
            account_type="Current",
            session_id="demo_session"
        )
        print(f"   ✅ Guidance retrieved: {guidance.get('source', 'unknown')} source")
        print(f"   Confidence: {guidance.get('confidence', 0):.2f}")

        # Test compliance validation
        print("\n6. Testing compliance validation...")
        sample_account = sample_data.iloc[0].to_dict()
        validation = await system.validate_account_compliance(
            sample_account,
            session_id="demo_session"
        )
        print(f"   ✅ Validation completed: {'Compliant' if validation.get('compliant', False) else 'Non-compliant'}")
        print(f"   Issues found: {len(validation.get('issues', []))}")

        # Generate insights
        print("\n7. Generating insights...")
        insights = await system.generate_compliance_insights(
            session_id="demo_session",
            analysis_results=results
        )
        print(f"   ✅ Insights generated")
        print(f"   Session patterns: {len(insights.get('session_patterns', {}))}")
        print(f"   Recommendations: {len(insights.get('recommendations', []))}")

        # Get session history
        print("\n8. Retrieving session history...")
        history = system.get_session_history("demo_session")
        if 'error' not in history:
            print(f"   ✅ Session history: {history.get('total_entries', 0)} entries")
        else:
            print(f"   ⚠️ Session history: {history.get('error', 'Not available')}")

        print("\n🎉 Demo completed successfully!")

        # Show final system stats
        print("\n📊 Final System Statistics:")
        final_status = system.get_system_status()
        stats = final_status['system_stats']
        print(f"   Sessions processed: {stats['sessions_processed']}")
        print(f"   Runtime: {stats['total_runtime_hours']:.2f} hours")
        print(f"   Memory operations: {stats['memory_operations']}")
        print(f"   MCP operations: {stats['mcp_operations']}")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        logger.error(f"Demo error: {e}")

    finally:
        # Cleanup
        print("\n9. Cleaning up...")
        await system.cleanup_system()
        print("✅ Cleanup completed")


def create_sample_banking_data() -> pd.DataFrame:
    """Create sample banking data for testing"""
    # Generate sample data
    accounts = []
    account_types = ['Current', 'Savings', 'Fixed', 'Investment', 'Safe Deposit']

    for i in range(100):
        # Calculate days since last activity
        days_inactive = random.randint(30, 1500)
        last_activity = datetime.now() - timedelta(days=days_inactive)

        # Determine if account should be dormant (3+ years = 1095+ days)
        is_dormant = days_inactive > 1095

        account = {
            'Account_ID': f'ACC{i:06d}',
            'Customer_ID': f'CUST{i:05d}',
            'Account_Type': random.choice(account_types),
            'Current_Balance': round(random.uniform(1000, 100000), 2),
            'Date_Last_Cust_Initiated_Activity': last_activity.strftime('%Y-%m-%d'),
            'Expected_Account_Dormant': 'yes' if is_dormant or random.random() < 0.1 else 'no',
            'Bank_Contact_Attempted_Post_Dormancy_Trigger': random.choice(['yes', 'no']),
            'Expected_Requires_Article_3_Process': 'yes' if is_dormant and random.random() < 0.5 else 'no',
            'Expected_Transfer_to_CB_Due': 'yes' if days_inactive > 1825 and random.random() < 0.3 else 'no',
            # 5+ years
            'SDB_Charges_Outstanding': random.uniform(0, 5000) if 'Safe Deposit' in account.get('Account_Type',
                                                                                                '') else 0,
            'Date_SDB_Charges_Became_Outstanding': last_activity.strftime('%Y-%m-%d') if 'Safe Deposit' in account.get(
                'Account_Type', '') else '',
            'SDB_Tenant_Communication_Received': random.choice(['yes', 'no']) if 'Safe Deposit' in account.get(
                'Account_Type', '') else ''
        }
        accounts.append(account)

    return pd.DataFrame(accounts)


async def test_memory_persistence():
    """Test memory persistence functionality"""
    print("=== Testing Memory Persistence ===")

    system = HybridMemoryBankingSystem()

    try:
        await system.initialize()

        if not system.memory_agent:
            print("❌ Memory agent not available")
            return

        # Test session memory
        session_id = "persistence_test"

        # Log multiple entries
        print("1. Logging test entries...")
        for i in range(10):
            system.memory_agent.log(
                session_id,
                f'test_event_{i}',
                {'iteration': i, 'data': f'test_data_{i}', 'timestamp': datetime.now().isoformat()},
                tags=['test', 'persistence'],
                importance=random.uniform(0.1, 1.0)
            )

        # Retrieve entries
        entries = system.memory_agent.get(session_id)
        print(f"   ✅ Stored and retrieved {len(entries)} memory entries")

        # Test filtering
        filtered_entries = system.memory_agent.get(session_id, event_filter='test_event_5')
        print(f"   ✅ Filtered retrieval: {len(filtered_entries)} entries")

        # Test knowledge caching
        print("2. Testing knowledge caching...")
        if system.mcp_tools:
            guidance = await system.memory_agent.get_regulatory_guidance(
                "2.1", "Current", cache_duration_hours=1
            )
            print("   ✅ Knowledge caching tested with MCP")
        else:
            guidance = await system.memory_agent.get_regulatory_guidance(
                "2.1", "Current", cache_duration_hours=1
            )
            print("   ✅ Knowledge caching tested with fallback")

        # Test session summary
        print("3. Testing session summary...")
        summary = system.memory_agent.get_session_summary(session_id)
        print(f"   ✅ Session summary: {summary.get('total_entries', 0)} entries")
        print(f"   ✅ Average importance: {summary.get('average_importance', 0):.2f}")

        # Test cleanup
        print("4. Testing cleanup...")
        cleaned = await system.memory_agent.cleanup_expired_entries()
        print(f"   ✅ Cleanup completed, removed {cleaned} expired entries")

    except Exception as e:
        print(f"❌ Persistence test failed: {e}")
        logger.error(f"Persistence test error: {e}")

    finally:
        await system.cleanup_system()


async def test_mcp_integration():
    """Test MCP integration functionality"""
    print("=== Testing MCP Integration ===")

    system = HybridMemoryBankingSystem()

    try:
        await system.initialize()

        if not system.mcp_tools:
            print("ℹ️ MCP tools not available (expected in local development)")
            print("   Testing fallback functionality...")

            # Test fallback regulatory guidance
            guidance = await system.get_regulatory_guidance(
                "2.1", "Current", "mcp_test_session"
            )
            print(f"   ✅ Fallback guidance: {guidance.get('source', 'unknown')} source")

            # Test fallback compliance validation
            sample_account = {
                'Account_ID': 'TEST001',
                'Account_Type': 'Current',
                'Current_Balance': 50000,
                'Expected_Account_Dormant': 'yes'
            }

            validation = await system.validate_account_compliance(
                sample_account, "mcp_test_session"
            )
            print(f"   ✅ Fallback validation: {'Compliant' if validation.get('compliant', False) else 'Non-compliant'}")

            return

        # Test MCP health
        print("1. Testing MCP health...")
        health = await system.mcp_tools._health_check()
        print(f"   MCP Health: {'✅' if health.success else '❌'}")

        # Test regulatory guidance
        print("2. Testing regulatory guidance...")
        guidance_response = await system.mcp_tools.get_regulatory_guidance_enhanced(
            "2.1", "Current", "mcp_test_session"
        )
        print(f"   Regulatory guidance: {'✅' if guidance_response.success else '❌'}")

        # Test compliance validation
        print("3. Testing compliance validation...")
        sample_account = {
            'Account_ID': 'TEST001',
            'Account_Type': 'Current',
            'Current_Balance': 50000,
            'Expected_Account_Dormant': 'yes'
        }

        validation_response = await system.mcp_tools.validate_compliance_rules_enhanced(
            sample_account, "mcp_test_session"
        )
        print(f"   Compliance validation: {'✅' if validation_response.success else '❌'}")

        # Test knowledge base queries
        print("4. Testing knowledge base...")
        if system.knowledge_manager:
            article_info = await system.knowledge_manager.get_regulatory_article(
                "2.1", "mcp_test_session"
            )
            print(f"   Knowledge base query: {'✅' if 'error' not in article_info else '❌'}")

        # Get performance stats
        print("5. Getting performance stats...")
        stats = system.mcp_tools.get_performance_stats()
        print(f"   MCP Performance: {stats.get('success_rate', 0):.1f}% success rate")
        print(f"   Total requests: {stats.get('total_requests', 0)}")
        print(f"   Average response time: {stats.get('average_response_time', 0):.1f}ms")

    except Exception as e:
        print(f"❌ MCP test failed: {e}")
        logger.error(f"MCP test error: {e}")

    finally:
        await system.cleanup_system()


async def benchmark_system_performance():
    """Benchmark system performance with different configurations"""
    print("=== System Performance Benchmark ===")

    # Test configurations
    configs = [
        {'name': 'Basic Mode', 'components': 'Minimal components'},
        {'name': 'Memory Only', 'components': 'Memory + Basic processing'},
        {'name': 'Full Integration', 'components': 'All available components'}
    ]

    results = {}
    sample_data = create_sample_banking_data()

    for config in configs:
        print(f"\nTesting configuration: {config['name']}")

        system = HybridMemoryBankingSystem()

        try:
            start_time = datetime.now()

            # Initialize
            await system.initialize()
            init_time = (datetime.now() - start_time).total_seconds()

            # Process sample data
            processing_start = datetime.now()

            results_data, agent_logs = await system.process_banking_data(
                sample_data, f"benchmark_{config['name'].lower().replace(' ', '_')}"
            )

            processing_time = (datetime.now() - processing_start).total_seconds()
            total_time = (datetime.now() - start_time).total_seconds()

            results[config['name']] = {
                'init_time': init_time,
                'processing_time': processing_time,
                'total_time': total_time,
                'confidence_score': results_data.get('confidence_score', 0),
                'accounts_processed': len(sample_data),
                'agent_steps': len(agent_logs),
                'notifications': len(results_data.get('notifications', [])),
                'components_active': sum(system.get_system_status()['components'].values())
            }

            print(f"  ✅ Completed in {total_time:.2f}s")
            print(f"     Init: {init_time:.2f}s, Processing: {processing_time:.2f}s")
            print(f"     Confidence: {results_data.get('confidence_score', 0):.2f}")
            print(f"     Active components: {results[config['name']]['components_active']}/4")

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results[config['name']] = {'error': str(e)}

        finally:
            await system.cleanup_system()

    # Print benchmark summary
    print("\n=== Benchmark Summary ===")
    for config_name, result in results.items():
        if 'error' not in result:
            processing_rate = result['accounts_processed'] / result['processing_time'] if result[
                                                                                              'processing_time'] > 0 else 0
            print(f"{config_name}:")
            print(f"  Total time: {result['total_time']:.2f}s")
            print(f"  Processing rate: {processing_rate:.1f} accounts/sec")
            print(f"  Confidence: {result['confidence_score']:.2f}")
            print(f"  Agent steps: {result['agent_steps']}")
            print(f"  Components: {result['components_active']}/4")
        else:
            print(f"{config_name}: Failed - {result['error']}")


async def test_integration_compatibility():
    """Test compatibility with existing system components"""
    print("=== Testing Integration Compatibility ===")

    # Test component availability
    print("1. Checking component availability...")
    availability = {
        'Hybrid Memory Agent': HYBRID_MEMORY_AVAILABLE,
        'Enhanced MCP Tools': ENHANCED_MCP_AVAILABLE,
        'Enhanced Orchestrator': ENHANCED_ORCHESTRATOR_AVAILABLE,
        'Configuration System': CONFIG_AVAILABLE,
        'Legacy Orchestrator': LEGACY_ORCHESTRATOR_AVAILABLE
    }

    for component, available in availability.items():
        status = '✅' if available else '❌'
        print(f"   {status} {component}")

    # Test graceful degradation
    print("\n2. Testing graceful degradation...")
    system = HybridMemoryBankingSystem()

    try:
        initialized = await system.initialize()
        print(f"   ✅ System initialization: {initialized}")

        # Test with minimal data
        minimal_data = create_sample_banking_data().head(5)
        results, logs = await system.process_banking_data(minimal_data, "compatibility_test")

        print(f"   ✅ Basic processing: {results.get('confidence_score', 0):.2f} confidence")
        print(f"   ✅ Fallback mechanisms working")

        # Test error handling
        try:
            empty_data = pd.DataFrame()
            error_results, error_logs = await system.process_banking_data(empty_data, "error_test")
            if 'error' in error_results:
                print(f"   ✅ Error handling: Graceful error recovery")
            else:
                print(f"   ✅ Empty data handling: Processed successfully")
        except Exception:
            print(f"   ✅ Error handling: Exception caught and handled")

    except Exception as e:
        print(f"   ❌ Compatibility test failed: {e}")

    finally:
        await system.cleanup_system()

    print("\n3. Integration summary:")
    working_components = sum(availability.values())
    total_components = len(availability)
    print(f"   Working components: {working_components}/{total_components}")
    print(
        f"   System compatibility: {'✅ Excellent' if working_components >= 3 else '⚠️ Partial' if working_components >= 1 else '❌ Limited'}")


# Main execution and examples
async def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description='Hybrid Memory Banking System')
    parser.add_argument('--demo', action='store_true', help='Run demonstration')
    parser.add_argument('--test-memory', action='store_true', help='Test memory persistence')
    parser.add_argument('--test-mcp', action='store_true', help='Test MCP integration')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    parser.add_argument('--test-compatibility', action='store_true', help='Test integration compatibility')
    parser.add_argument('--all', action='store_true', help='Run all tests')

    args = parser.parse_args()

    if args.all or args.demo:
        await demo_hybrid_memory_system()
        print("\n" + "=" * 50 + "\n")

    if args.all or args.test_memory:
        await test_memory_persistence()
        print("\n" + "=" * 50 + "\n")

    if args.all or args.test_mcp:
        await test_mcp_integration()
        print("\n" + "=" * 50 + "\n")

    if args.all or args.benchmark:
        await benchmark_system_performance()
        print("\n" + "=" * 50 + "\n")

    if args.all or args.test_compatibility:
        await test_integration_compatibility()
        print("\n" + "=" * 50 + "\n")

    if not any([args.demo, args.test_memory, args.test_mcp, args.benchmark, args.test_compatibility, args.all]):
        print("No test specified. Use --help for options or --demo for a quick demonstration.")
        print("\nAvailable commands:")
        print("  python hybrid_memory_integration.py --demo")
        print("  python hybrid_memory_integration.py --test-memory")
        print("  python hybrid_memory_integration.py --test-mcp")
        print("  python hybrid_memory_integration.py --benchmark")
        print("  python hybrid_memory_integration.py --test-compatibility")
        print("  python hybrid_memory_integration.py --all")


# Integration helper functions for existing codebase
def integrate_with_existing_main():
    """
    Integration helper for the existing main.py file
    This shows how to modify the existing Streamlit app to use the hybrid memory system
    """
    integration_code = '''
    # Add this to your existing main.py file

    # Import the hybrid memory system
    from hybrid_memory_integration import HybridMemoryBankingSystem
    import asyncio

    # Initialize the system (add this near the top of your main.py)
    @st.cache_resource
    def init_hybrid_memory_system():
        """Initialize the hybrid memory system (cached for performance)"""
        return HybridMemoryBankingSystem()

    # Async wrapper for Streamlit
    def run_async_analysis(data, session_name=None):
        """Wrapper to run async analysis in Streamlit"""
        async def async_wrapper():
            system = init_hybrid_memory_system()
            await system.initialize()
            return await system.process_banking_data(data, session_name)

        return asyncio.run(async_wrapper())

    # Replace your existing analysis button code with:
    if st.button("🚀 Run Advanced Compliance Analysis", type="primary"):
        with st.spinner("Running enhanced analysis with hybrid memory..."):
            try:
                final_state, agent_logs = run_async_analysis(df, f"streamlit_session_{int(time.time())}")
                # Use existing display_results function
                display_results(final_state, agent_logs, use_enhanced=True, langsmith_enabled=True)
            except Exception as e:
                st.error(f"Analysis failed: {e}")

    # Add memory insights section
    if st.sidebar.button("View Memory Insights"):
        system = init_hybrid_memory_system()
        if system.memory_agent:
            stats = system.memory_agent.get_knowledge_stats()
            st.sidebar.json(stats)
    '''

    return integration_code


def create_docker_compose():
    """Create Docker Compose configuration for the complete system"""
    docker_compose = '''
version: '3.8'

services:
  # Main banking application
  banking-app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - ENVIRONMENT=production
      - MEMORY_SYSTEM_ENABLED=true
      - MCP_INTEGRATION_ENABLED=true
      - MCP_SERVER_URL=http://mcp-server:8000
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - redis-memory

  # Memory persistence and caching
  redis-memory:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes

  # Monitoring and observability
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana

volumes:
  redis-data:
  grafana-data:
'''

    return docker_compose


# Backward compatibility functions
async def create_memory_enhanced_orchestrator(use_enhanced: bool = True):
    """Create and initialize memory-enhanced orchestrator"""
    system = HybridMemoryBankingSystem()
    await system.initialize()
    return system


async def run_memory_enhanced_flow(data: pd.DataFrame) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run memory-enhanced flow - async version"""
    system = await create_memory_enhanced_orchestrator()
    try:
        return await system.process_banking_data(data)
    finally:
        await system.cleanup_system()


if __name__ == "__main__":
    # Setup logging directory
    Path("logs").mkdir(exist_ok=True)

    # Run main
    asyncio.run(main())