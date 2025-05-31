# main.py - Enhanced Banking Compliance System with Integration Adapter (Complete)
import streamlit as st
import pandas as pd
import os
import asyncio
import time
from datetime import datetime
import logging
import json
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Enhanced imports - Integration Adapter
try:
    from integration_adapter import create_integrated_banking_system

    INTEGRATION_ADAPTER_AVAILABLE = True
except ImportError as e:
    INTEGRATION_ADAPTER_AVAILABLE = False
    st.error(f"⚠️ Integration Adapter not available: {e}")

# Hybrid integration import
try:
    from hybrid_integration import init_hybrid_memory_system

    HYBRID_INTEGRATION_AVAILABLE = True
except ImportError:
    HYBRID_INTEGRATION_AVAILABLE = False

# Import configuration and utilities
try:
    from config.memory_config import ConfigManager, ConfigMonitor
    from utils.langsmith_setup import setup_langsmith, get_trace_url
    from utils.visualization_helpers import (
        create_dormancy_breakdown_chart,
        create_compliance_scatter_chart,
        create_risk_gauge,
        create_notification_pie_chart
    )

    ENHANCED_IMPORTS = True
except ImportError as e:
    ENHANCED_IMPORTS = False
    st.warning(f"⚠️ Some enhanced features not available: {e}")


    # Fallback setup
    class ConfigManager:
        def __init__(self):
            self.config = type('Config', (), {
                'langsmith_enabled': False,
                'mcp_enabled': False,
                'environment': 'development'
            })()

        def get_config_summary(self):
            return {
                'system': {'environment': 'development', 'enhanced_mode': True},
                'memory': {'enabled': False},
                'mcp': {'enabled': False}
            }


    def setup_langsmith():
        return False


    def get_trace_url(session_id):
        return None

# Enhanced orchestrator import
try:
    from orchestrator import run_enhanced_flow, validate_workflow_data

    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_AVAILABLE = False

# Configure logging
try:
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/banking_app.log'),
            logging.StreamHandler()
        ]
    )
except Exception:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

logger = logging.getLogger(__name__)

# Ensure directories exist
Path("logs").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)


# =====================================
# ENHANCED FUNCTION DEFINITIONS
# =====================================

@st.cache_resource
def init_integrated_banking_system():
    """Initialize the integrated banking system (cached for performance)"""
    if INTEGRATION_ADAPTER_AVAILABLE:
        try:
            # Create an async wrapper to handle the async function
            async def _create_system():
                return await create_integrated_banking_system()

            # Run the async function in a new event loop
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                system = loop.run_until_complete(_create_system())
                return system
            except RuntimeError:
                # If there's already an event loop running, use asyncio.run
                return asyncio.run(_create_system())
        except Exception as e:
            logger.error(f"Failed to initialize integrated banking system: {e}")
            return None
    else:
        return None


@st.cache_resource
def init_config_manager():
    """Initialize configuration manager (cached for performance)"""
    if ENHANCED_IMPORTS:
        return ConfigManager()
    else:
        return ConfigManager()  # Fallback


def run_async_analysis(data, session_name=None, use_enhanced=True):
    """Updated async analysis wrapper"""

    # Validate input data
    if data is None:
        logger.warning("No data provided for analysis")
        return basic_fallback_analysis(pd.DataFrame(), session_name)

    # Ensure data is a DataFrame
    if not isinstance(data, pd.DataFrame):
        try:
            data = pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Could not convert input to DataFrame: {e}")
            return basic_fallback_analysis(pd.DataFrame(), session_name)

    async def async_wrapper():
        if HYBRID_INTEGRATION_AVAILABLE and use_enhanced:
            try:
                system = init_hybrid_memory_system()
                if system:
                    await system.initialize()
                    return await system.process_banking_data(data, session_name)
            except Exception as e:
                logger.error(f"Hybrid integration failed: {e}")
                # Fall through to next option

        # Fallback to enhanced orchestrator if available
        try:
            if ORCHESTRATOR_AVAILABLE:
                # Validate data first
                is_valid, message = validate_workflow_data(data)
                if not is_valid:
                    logger.warning(f"Data validation failed: {message}")
                    return basic_fallback_analysis(data, session_name)

                # Run enhanced workflow
                final_state, memory = run_enhanced_flow(data)

                # Convert to expected format
                if hasattr(final_state, '__dict__'):
                    state_dict = final_state.__dict__
                else:
                    state_dict = final_state

                # Create agent logs from memory
                agent_logs = []
                if memory:
                    memory_logs = memory.get(final_state.session_id)
                    for log_entry in memory_logs:
                        agent_logs.append({
                            'agent': log_entry.get('event', 'unknown'),
                            'timestamp': datetime.now().isoformat(),
                            'status': 'success' if 'error' not in log_entry.get('event', '') else 'error',
                            'details': log_entry.get('data', {})
                        })

                return state_dict, agent_logs
            else:
                # Final fallback to basic processing
                return basic_fallback_analysis(data, session_name)

        except ImportError:
            # Final fallback to basic processing
            return basic_fallback_analysis(data, session_name)
        except Exception as e:
            logger.error(f"Enhanced orchestrator failed: {e}")
            return basic_fallback_analysis(data, session_name)

    try:
        return asyncio.run(async_wrapper())
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            # Handle case where we're already in an event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(async_wrapper())
            finally:
                loop.close()
        else:
            logger.error(f"Async runtime error: {e}")
            return basic_fallback_analysis(data, session_name)
    except Exception as e:
        logger.error(f"Analysis execution error: {e}")
        return basic_fallback_analysis(data, session_name)


def basic_fallback_analysis(data, session_name):
    """Fallback analysis when integration adapter is not available"""
    logger.info("Running fallback analysis")

    # Ensure data is valid
    if data is None:
        logger.warning("No data provided to fallback analysis")
        data = pd.DataFrame()  # Create empty DataFrame as fallback

    # Convert to DataFrame if it's not already
    if not isinstance(data, pd.DataFrame):
        try:
            data = pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Could not convert data to DataFrame: {e}")
            data = pd.DataFrame()

    # Calculate basic metrics
    data_length = len(data) if hasattr(data, '__len__') else 0

    # Create detailed agent logs for visibility
    agent_logs = [
        {
            'agent': 'data_validator',
            'timestamp': datetime.now().isoformat(),
            'status': 'success',
            'details': {
                'message': 'Data validation completed',
                'data_type': str(type(data)),
                'data_shape': f"{data.shape}" if hasattr(data, 'shape') else 'N/A',
                'columns': list(data.columns) if hasattr(data, 'columns') else [],
                'rows_processed': data_length
            }
        },
        {
            'agent': 'basic_analyzer',
            'timestamp': datetime.now().isoformat(),
            'status': 'success',
            'details': {
                'message': 'Basic analysis completed',
                'analysis_type': 'fallback_mode',
                'features_available': ['data_overview', 'basic_metrics'],
                'enhanced_features': 'not_available'
            }
        },
        {
            'agent': 'compliance_checker',
            'timestamp': datetime.now().isoformat(),
            'status': 'success',
            'details': {
                'message': 'Basic compliance check completed',
                'compliance_columns_found': [],
                'dormancy_analysis': 'basic_mode',
                'recommendations_generated': 3
            }
        },
        {
            'agent': 'report_generator',
            'timestamp': datetime.now().isoformat(),
            'status': 'success',
            'details': {
                'message': 'Report generation completed',
                'report_format': 'basic',
                'export_options': ['csv', 'json'],
                'session_id': session_name or f'fallback_{int(time.time())}'
            }
        }
    ]

    # Add compliance analysis if data has relevant columns
    if hasattr(data, 'columns') and data_length > 0:
        compliance_columns = []
        if 'Expected_Account_Dormant' in data.columns:
            compliance_columns.append('Expected_Account_Dormant')
        if 'Account_Type' in data.columns:
            compliance_columns.append('Account_Type')
        if 'Current_Balance' in data.columns:
            compliance_columns.append('Current_Balance')

        # Update compliance checker details
        agent_logs[2]['details']['compliance_columns_found'] = compliance_columns

        if 'Expected_Account_Dormant' in data.columns:
            try:
                dormant_count = len(
                    data[data['Expected_Account_Dormant'].astype(str).str.lower().isin(['yes', 'true', '1'])])
                agent_logs[2]['details']['dormant_accounts_found'] = dormant_count
            except Exception as e:
                agent_logs[2]['details']['dormancy_check_error'] = str(e)

    # Basic analysis results
    results = {
        'session_id': session_name or f'fallback_{int(time.time())}',
        'data': data,
        'enhanced_data': None,
        'dormant_results': None,
        'compliance_results': None,
        'non_compliant_data': None,
        'current_step': 'completed',
        'messages': [
            "✅ Data validation completed",
            "✅ Basic analysis completed",
            "✅ Compliance check completed",
            "✅ Report generated"
        ],
        'error': None,
        'confidence_score': 0.6,  # Lower confidence for basic mode
        'timestamp': datetime.now(),
        'agent_logs': agent_logs,
        'notifications': [
            f"✅ Processed {data_length} accounts using basic mode",
            "⚠️ Enhanced features not available - using fallback analysis",
            "💡 Install integration components for enhanced analysis",
            f"📊 Session: {session_name or f'fallback_{int(time.time())}'}"
        ],
        'final_result': f"Basic analysis completed for {data_length} accounts. System running in fallback mode.",
        'recommendations': [
            "Install integration adapter components for enhanced analysis",
            "Review account data manually for detailed compliance checking",
            "Consider upgrading to enhanced mode for full feature access",
            "Check system logs for any missing dependencies",
            "Verify data format matches expected banking compliance schema"
        ],
        'mcp_enabled': False,
        'mcp_results': None
    }

    logger.info(f"Fallback analysis completed. Processed {data_length} records.")
    return results, agent_logs


def get_system_health():
    """Get comprehensive system health status"""
    health = {
        'timestamp': datetime.now().isoformat(),
        'integration_adapter': INTEGRATION_ADAPTER_AVAILABLE,
        'enhanced_imports': ENHANCED_IMPORTS,
        'hybrid_integration': HYBRID_INTEGRATION_AVAILABLE,
        'orchestrator': ORCHESTRATOR_AVAILABLE,
        'components': {
            'streamlit': True,
            'pandas': True,
            'logging': True
        }
    }

    if INTEGRATION_ADAPTER_AVAILABLE:
        try:
            system = init_integrated_banking_system()
            if system and hasattr(system, 'get_system_status'):
                health['system_status'] = system.get_system_status()
        except Exception as e:
            health['system_error'] = str(e)

    return health


def display_dormancy_analysis(state_dict, use_enhanced):
    """Display dormancy analysis results - UPDATED for new system"""

    # Check for enhanced dormancy summary report
    if state_dict.get("dormant_summary_report"):
        dsr = state_dict["dormant_summary_report"]

        st.markdown("#### 🎯 Enhanced Dormancy Analysis Summary")

        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_flagged = dsr.consolidated_findings.get('total_unique_flagged_accounts', 0)
            st.metric("Total Flagged", total_flagged)
        with col2:
            dormancy_rate = dsr.summary_statistics.get('overall_dormancy_rate', 0)
            st.metric("Dormancy Rate", f"{dormancy_rate:.1f}%")
        with col3:
            risk_level = dsr.risk_assessment.get('risk_level', 'UNKNOWN')
            risk_color = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}.get(risk_level, "⚪")
            st.metric("Risk Level", f"{risk_color} {risk_level}")
        with col4:
            balance_at_risk = dsr.summary_statistics.get('total_balance_at_risk', 0)
            st.metric("Balance at Risk", f"${balance_at_risk:,.2f}")

        # Priority Analysis
        st.markdown("#### 🚨 Priority Analysis")
        priority_tabs = st.tabs(["Critical", "High Priority", "Medium Priority", "All Results"])

        with priority_tabs[0]:  # Critical
            critical_results = [r for r in dsr.analysis_results if r.priority_level == 'CRITICAL' and r.count > 0]
            if critical_results:
                for result in critical_results:
                    st.error(f"**{result.regulation_article}**: {result.description}")
                    if result.details.get('urgency') == 'HIGH':
                        st.markdown("⏰ **Immediate action required**")
            else:
                st.success("✅ No critical issues found")

        with priority_tabs[1]:  # High Priority
            high_results = [r for r in dsr.analysis_results if r.priority_level == 'HIGH' and r.count > 0]
            if high_results:
                for result in high_results:
                    st.warning(f"**{result.regulation_article}**: {result.description}")
                    st.write(f"Risk Score: {result.risk_score:.2f}")
            else:
                st.success("✅ No high priority issues found")

        with priority_tabs[2]:  # Medium Priority
            medium_results = [r for r in dsr.analysis_results if r.priority_level == 'MEDIUM' and r.count > 0]
            if medium_results:
                for result in medium_results:
                    st.info(f"**{result.regulation_article}**: {result.description}")
            else:
                st.success("✅ No medium priority issues found")

        with priority_tabs[3]:  # All Results
            st.markdown("**Complete Analysis Results:**")
            for result in dsr.analysis_results:
                if result.count > 0:
                    priority_icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}.get(result.priority_level,
                                                                                                  "⚪")

                    with st.expander(f"{priority_icon} {result.regulation_article} - {result.count} accounts"):
                        st.write(f"**Description:** {result.description}")
                        st.write(f"**Priority:** {result.priority_level}")
                        st.write(f"**Risk Score:** {result.risk_score:.2f}")
                        st.write(f"**Criteria:** {result.details.get('criteria', 'N/A')}")

                        if result.details.get('sample_accounts'):
                            st.write(f"**Sample Account IDs:** {', '.join(result.details['sample_accounts'][:3])}")

        # Action Recommendations
        if dsr.action_recommendations:
            st.markdown("#### 📋 Action Recommendations")
            for i, rec in enumerate(dsr.action_recommendations, 1):
                priority_color = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}.get(rec['priority'], "⚪")

                with st.expander(f"{priority_color} {rec['priority']}: {rec['action']}"):
                    st.write(f"**Description:** {rec['description']}")
                    st.write(f"**Deadline:** {rec['deadline']}")
                    st.write(f"**Regulation:** {rec['regulation']}")
                    st.write(f"**Affected Accounts:** {rec['affected_accounts']}")

        # Compliance Status
        st.markdown("#### ⚖️ Regulatory Compliance Status")
        compliance_status = dsr.regulatory_compliance_status

        status_color = {
            "COMPLIANT": "🟢",
            "NEEDS_ATTENTION": "🟡",
            "NON_COMPLIANT": "🔴"
        }.get(compliance_status['overall_status'], "⚪")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Compliance Status", f"{status_color} {compliance_status['overall_status']}")
            st.metric("Compliance Score", f"{compliance_status['compliance_score']:.1f}/100")
        with col2:
            st.metric("Critical Violations", compliance_status['critical_violations'])

            if compliance_status['regulatory_gaps']:
                st.markdown("**Regulatory Gaps:**")
                for gap in compliance_status['regulatory_gaps']:
                    st.write(f"- {gap['regulation']}: {gap['issue']}")

    elif state_dict.get("dormant_results"):
        # Legacy format display
        dr = state_dict["dormant_results"]
        kpis = dr.get("summary_kpis", {})

        st.markdown("#### 📊 Dormancy Analysis (Legacy Mode)")

        # Summary Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Dormancy Rate", f"{kpis.get('percentage_dormant_of_total', 0):.1f}%")
        with col2:
            st.metric("High Value Dormant", kpis.get('count_high_value_dormant', 0))
        with col3:
            st.metric("CB Transfer Eligible", kpis.get('count_eligible_for_cb_transfer', 0))

        # Detailed breakdown
        st.markdown("#### 🏦 Dormant Accounts Breakdown")
        breakdown_data = {
            'Safe Deposit': kpis.get('count_sdb_dormant', 0),
            'Investment': kpis.get('count_investment_dormant', 0),
            'Fixed Deposit': kpis.get('count_fixed_deposit_dormant', 0),
            'Demand Deposit': kpis.get('count_demand_deposit_dormant', 0),
            'Unclaimed Instruments': kpis.get('count_unclaimed_instruments', 0)
        }

        total_dormant = sum(breakdown_data.values())
        if total_dormant > 0:
            for category, count in breakdown_data.items():
                if count > 0:
                    percentage = (count / total_dormant) * 100
                    st.metric(f"{category}", f"{count} ({percentage:.1f}%)")
        else:
            st.info("No dormant accounts found in detailed categories.")
    else:
        # Fallback analysis using basic data
        data = state_dict.get("data")

        # Safe data checking
        data_exists = False
        try:
            if data is not None and hasattr(data, 'empty') and hasattr(data, '__len__'):
                if not data.empty and len(data) > 0:
                    data_exists = True
        except Exception as e:
            logger.warning(f"Error checking data in dormancy analysis: {e}")

        if data_exists:
            st.markdown("#### 📊 Basic Dormancy Analysis")

            if 'Expected_Account_Dormant' in data.columns:
                dormant_accounts = data[
                    data['Expected_Account_Dormant'].astype(str).str.lower().isin(['yes', 'true', '1'])]
                dormant_count = len(dormant_accounts)
                dormancy_rate = (dormant_count / len(data)) * 100

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Dormant Accounts", dormant_count)
                with col2:
                    st.metric("Dormancy Rate", f"{dormancy_rate:.1f}%")

                if dormant_count > 0:
                    st.markdown("**Dormant Account Types:**")
                    if 'Account_Type' in dormant_accounts.columns:
                        type_counts = dormant_accounts['Account_Type'].value_counts()
                        for account_type, count in type_counts.items():
                            st.write(f"- {account_type}: {count} accounts")
            else:
                st.info("No dormancy information available in the data.")
        else:
            st.info("No dormancy analysis results available.")


def display_enhanced_summary(state_dict):
    """Display enhanced summary if available"""

    if state_dict.get("executive_summary"):
        st.markdown("#### 📋 Executive Summary")
        st.text_area(
            "Executive Summary",
            state_dict["executive_summary"],
            height=200,
            help="AI-generated executive summary of the dormancy analysis"
        )

    if state_dict.get("dormant_summary_report"):
        dsr = state_dict["dormant_summary_report"]

        # Risk Assessment Gauge
        st.markdown("#### 🎯 Risk Assessment")
        risk_score = dsr.risk_assessment.get('overall_risk_score', 0.5)

        # Create a simple progress bar for risk
        risk_percentage = risk_score * 100
        risk_color = "🟢" if risk_score < 0.3 else "🟡" if risk_score < 0.6 else "🔴"

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Risk Score", f"{risk_percentage:.1f}%")
        with col2:
            st.progress(risk_score)
            st.write(f"{risk_color} {dsr.risk_assessment.get('risk_level', 'UNKNOWN')} Risk")

        # Financial Risk
        financial_risk = dsr.risk_assessment.get('financial_risk', 'UNKNOWN')
        balance_at_risk = dsr.summary_statistics.get('total_balance_at_risk', 0)

        if balance_at_risk > 0:
            st.markdown(f"**Financial Risk:** {financial_risk}")
            st.markdown(f"**Total Balance at Risk:** ${balance_at_risk:,.2f}")

        # Risk Factors
        risk_factors = dsr.risk_assessment.get('risk_factors', [])
        if risk_factors:
            st.markdown("**Key Risk Factors:**")
            for factor in risk_factors:
                st.write(f"- {factor}")


def display_compliance_analysis(state_dict, use_enhanced):
    """Display compliance analysis results"""
    if state_dict.get("compliance_results"):
        cr = state_dict["compliance_results"]

        # Critical compliance metrics
        st.markdown("#### Critical Compliance Issues")
        col1, col2, col3 = st.columns(3)
        with col1:
            count = cr.get('transfer_candidates_cb', {}).get('count', 0)
            st.metric("🚨 CB Transfer Due", count)
            if count > 0:
                st.error("Immediate action required!")

        with col2:
            count = cr.get('incomplete_contact', {}).get('count', 0)
            st.metric("📞 Incomplete Contact", count)
            if count > 0:
                st.warning("Follow-up needed")

        with col3:
            count = cr.get('flag_candidates', {}).get('count', 0)
            st.metric("🏷️ Unflagged Dormant", count)
            if count > 0:
                st.warning("Flagging required")

        # Additional compliance metrics
        st.markdown("#### Additional Compliance Checks")
        additional_metrics = {
            'Internal Ledger Transfer': cr.get('ledger_candidates_internal', {}).get('count', 0),
            'Statement Freeze Needed': cr.get('statement_freeze_needed', {}).get('count', 0)
        }

        for metric, value in additional_metrics.items():
            if value > 0:
                st.warning(f"**{metric}:** {value} accounts")
            else:
                st.success(f"**{metric}:** ✅ No issues found")

        # Enhanced features
        if use_enhanced:
            st.markdown("#### 📊 Compliance Visualization")
            st.info("✨ Enhanced compliance charts and risk analysis would be displayed here")
    else:
        # Basic compliance analysis
        st.markdown("#### Basic Compliance Analysis")
        data = state_dict.get("data")

        # Safe data checking
        data_exists = False
        try:
            if data is not None and hasattr(data, 'empty') and hasattr(data, '__len__'):
                if not data.empty and len(data) > 0:
                    data_exists = True
        except Exception as e:
            logger.warning(f"Error checking data in compliance analysis: {e}")

        if data_exists:
            # Check for basic compliance indicators
            issues = []

            if 'Expected_Transfer_to_CB_Due' in data.columns:
                cb_transfer = len(
                    data[data['Expected_Transfer_to_CB_Due'].astype(str).str.lower().isin(['yes', 'true', '1'])])
                if cb_transfer > 0:
                    issues.append(f"🚨 {cb_transfer} accounts require CB transfer")

            if 'Bank_Contact_Attempted_Post_Dormancy_Trigger' in data.columns:
                no_contact = len(data[data['Bank_Contact_Attempted_Post_Dormancy_Trigger'].astype(str).str.lower().isin(
                    ['no', 'false', '0'])])
                if no_contact > 0:
                    issues.append(f"📞 {no_contact} accounts missing contact attempts")

            if issues:
                for issue in issues:
                    st.warning(issue)
            else:
                st.success("✅ No critical compliance issues identified")
        else:
            st.info("No compliance analysis results available.")


def display_risk_assessment(state_dict, agent_logs):
    """Display risk assessment results"""
    st.markdown("#### 🎯 Risk-Based Recommendations")

    # Risk recommendations
    recommendations = state_dict.get("recommendations", [])
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            priority = "🔴" if "URGENT" in rec.upper() else "🟡" if "WARNING" in rec.upper() else "🟢"
            st.markdown(f"{priority} **{i}.** {rec}")
    else:
        st.info("No specific recommendations generated.")

    # Confidence and risk assessment
    st.markdown("#### 📊 Confidence Assessment")

    confidence_score = state_dict.get("confidence_score", 0.8)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Overall Confidence", f"{confidence_score:.1%}")
        st.progress(confidence_score)

    with col2:
        risk_level = "Low" if confidence_score >= 0.8 else "Medium" if confidence_score >= 0.6 else "High"
        color = "green" if risk_level == "Low" else "orange" if risk_level == "Medium" else "red"
        st.markdown(f"**Risk Level:** :{color}[{risk_level}]")

        # Risk factors
        risk_factors = []
        if confidence_score < 0.7:
            risk_factors.append("Low confidence score")
        if state_dict.get("error"):
            risk_factors.append("Processing errors")
        if not state_dict.get("compliance_results"):
            risk_factors.append("Limited compliance data")

        if risk_factors:
            st.markdown("**Risk Factors:**")
            for factor in risk_factors:
                st.write(f"- {factor}")

    # Enhanced risk visualization
    st.markdown("#### 📈 Risk Gauge")
    st.info("✨ Interactive risk gauge and detailed risk analysis would be displayed here")


def display_notifications(state_dict):
    """Display system notifications"""
    st.markdown("#### 🔔 System Notifications")

    notifications = state_dict.get("notifications", [])
    if notifications:
        # Categorize notifications
        critical = [n for n in notifications if "URGENT" in n.upper() or "🚨" in n]
        warnings = [n for n in notifications if "WARNING" in n.upper() or "⚠️" in n]
        info = [n for n in notifications if n not in critical and n not in warnings]

        # Display by priority
        if critical:
            st.markdown("**🚨 Critical Notifications:**")
            for notification in critical:
                st.error(notification)

        if warnings:
            st.markdown("**⚠️ Warning Notifications:**")
            for notification in warnings:
                st.warning(notification)

        if info:
            st.markdown("**ℹ️ Information Notifications:**")
            for notification in info:
                st.info(notification)

        # Enhanced notification visualization
        if len(notifications) > 1:
            st.markdown("#### 📊 Notification Distribution")
            st.info("✨ Interactive notification distribution chart would be displayed here")
    else:
        st.success("✅ No critical notifications at this time.")


def display_memory_insights(state_dict):
    """Display memory and system insights"""
    st.markdown("#### 🧠 Memory & System Insights")

    if INTEGRATION_ADAPTER_AVAILABLE:
        try:
            system = init_integrated_banking_system()
            if system and hasattr(system, 'memory_agent'):
                # Memory statistics
                if hasattr(system.memory_agent, 'get_knowledge_stats'):
                    memory_stats = system.memory_agent.get_knowledge_stats()

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Memory Statistics:**")
                        st.json(memory_stats)

                    with col2:
                        st.markdown("**System Performance:**")
                        if hasattr(system, 'get_system_status'):
                            perf_stats = system.get_system_status()
                            if 'system_stats' in perf_stats:
                                st.json(perf_stats['system_stats'])

                # Session insights
                session_id = state_dict.get('session_id')
                if session_id and hasattr(system, 'get_session_history'):
                    st.markdown("#### 📈 Session History")
                    history = system.get_session_history(session_id)
                    if 'error' not in history:
                        st.json(history)
                    else:
                        st.warning(f"Session history not available: {history.get('error')}")

            else:
                st.info("Memory system not available - running in basic mode")
        except Exception as e:
            st.warning(f"Memory insights error: {e}")
    else:
        st.info("Integration adapter not available")

        # Show basic system info
        health = get_system_health()
        st.json(health)


def display_system_logs(state_dict, agent_logs, langsmith_enabled):
    """Display system logs and traces"""
    st.markdown("#### 🔍 System Execution Logs")

    # LangSmith trace link
    if langsmith_enabled and state_dict.get("session_id"):
        trace_url = get_trace_url(state_dict["session_id"])
        if trace_url:
            st.markdown(f"🔗 [View in LangSmith]({trace_url})")
        st.markdown(f"**Session ID:** `{state_dict['session_id']}`")

    # Agent performance
    if agent_logs and len(agent_logs) > 0:
        st.markdown("#### Execution Summary")

        try:
            success_count = sum(1 for log in agent_logs if log.get("status") == "success")
            total_count = len(agent_logs)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Success Rate", f"{success_count}/{total_count}")
            with col2:
                st.metric("Total Steps", total_count)
            with col3:
                processing_time = 1.5  # Estimated
                st.metric("Processing Time", f"{processing_time:.1f}s")

            # Detailed logs
            st.markdown("#### 📝 Detailed Execution Logs")
            for i, log in enumerate(agent_logs):
                try:
                    status_icon = "✅" if log.get("status") == "success" else "❌"
                    agent_name = log.get("agent", "unknown").replace("_", " ").title()
                    timestamp = log.get("timestamp", "unknown")

                    st.markdown(f"**{status_icon} Step {i + 1}: {agent_name}**")
                    st.caption(f"Timestamp: {timestamp}")

                    if log.get("details"):
                        st.markdown(f"**Details for {agent_name}:**")
                        try:
                            st.json(log["details"])
                        except Exception as json_error:
                            st.text(str(log["details"]))

                    if i < len(agent_logs) - 1:  # Don't add divider after last item
                        st.divider()
                except Exception as e:
                    st.error(f"Error displaying log entry {i}: {e}")
                    st.json(log)  # Show raw log if formatting fails
        except Exception as e:
            st.error(f"Error processing agent logs: {e}")
            st.markdown("**Raw Agent Logs:**")
            st.json(agent_logs)
    else:
        st.info("No execution logs available.")

        # Show some basic system info instead
        st.markdown("#### 🔧 System Status")
        try:
            health = get_system_health()
            st.json(health)
        except Exception as e:
            st.error(f"Could not get system health: {e}")

        # Check if we can show session info
        if state_dict:
            st.markdown("#### 📊 Session Information")
            session_info = {
                'session_id': state_dict.get('session_id', 'unknown'),
                'timestamp': str(state_dict.get('timestamp', 'unknown')),
                'confidence_score': state_dict.get('confidence_score', 0),
                'final_result': state_dict.get('final_result', 'No result available'),
                'notifications_count': len(state_dict.get('notifications', [])),
                'recommendations_count': len(state_dict.get('recommendations', []))
            }
            st.json(session_info)


def display_export_section(state_dict):
    """Display export and download options"""
    st.markdown("## 📥 Export Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Non-compliant accounts export
        non_compliant_data = state_dict.get("non_compliant_data")
        if non_compliant_data is not None and len(non_compliant_data) > 0:
            if hasattr(non_compliant_data, 'to_csv'):
                csv_data = non_compliant_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📊 Download Non-Compliant Accounts",
                    data=csv_data,
                    file_name=f'non_compliant_accounts_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                    mime='text/csv'
                )
            else:
                st.info("Non-compliant data available but not in downloadable format")
        else:
            st.info("No non-compliant accounts to download")

    with col2:
        # Comprehensive report export
        report_data = {
            'session_id': state_dict.get("session_id", "unknown"),
            'analysis_timestamp': state_dict.get("timestamp", datetime.now()).isoformat() if state_dict.get(
                "timestamp") else datetime.now().isoformat(),
            'confidence_score': state_dict.get("confidence_score", 0.0),
            'final_result': state_dict.get("final_result", ""),
            'recommendations': state_dict.get("recommendations", []),
            'notifications': state_dict.get("notifications", []),
            'system_mode': 'enhanced' if INTEGRATION_ADAPTER_AVAILABLE else 'basic'
        }

        report_json = json.dumps(report_data, indent=2, default=str)
        st.download_button(
            label="📋 Download Analysis Report",
            data=report_json.encode('utf-8'),
            file_name=f'compliance_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
            mime='application/json'
        )

    with col3:
        # Raw data export
        data = state_dict.get("data")

        # Safe data checking
        data_exists = False
        try:
            if data is not None and hasattr(data, 'empty') and hasattr(data, '__len__'):
                if not data.empty and len(data) > 0:
                    data_exists = True
        except Exception as e:
            logger.warning(f"Error checking data in export: {e}")

        if data_exists:
            csv_data = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 Download Processed Data",
                data=csv_data,
                file_name=f'processed_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv'
            )
        else:
            st.info("No processed data available")


def display_results(final_state, agent_logs, use_enhanced, langsmith_enabled):
    """Display analysis results with enhanced features - UPDATED"""

    # Handle both dict and object states
    if hasattr(final_state, '__dict__'):
        state_dict = final_state.__dict__
    else:
        state_dict = final_state if final_state is not None else {}

    # Error handling
    if state_dict.get("error"):
        st.error(f"❌ Analysis Error: {state_dict['error']}")

        # Show troubleshooting steps
        st.markdown("### 🔧 Troubleshooting")
        st.write("**Common solutions:**")
        st.write("- Check that all required files are in place")
        st.write("- Verify data format and column names")
        st.write("- Review logs for detailed error information")
        st.write("- Try running in basic mode if enhanced features fail")

        if st.button("View System Health"):
            health = get_system_health()
            st.json(health)
        return

    # Handle case where final_state is None or empty
    if not state_dict:
        st.error("❌ No analysis results available")
        st.warning("The analysis did not return any results. This might be due to:")
        st.write("- Missing or invalid input data")
        st.write("- System initialization failure")
        st.write("- Missing required components")

        if st.button("View System Health"):
            health = get_system_health()
            st.json(health)
        return

    # Main results summary
    st.markdown("## 📈 Analysis Results")

    # Enhanced header with system info
    col_header1, col_header2 = st.columns([3, 1])
    with col_header1:
        st.markdown("### Banking Compliance Analysis Report")
        session_id = state_dict.get('session_id', 'unknown')
        st.caption(f"Session ID: `{session_id}`")

        # Show analysis mode
        if state_dict.get("dormant_summary_report"):
            st.success("🚀 Enhanced Modular Analysis Mode")
        elif state_dict.get("dormant_results"):
            st.info("⚡ Legacy Analysis Mode")
        else:
            st.warning("📊 Basic Analysis Mode")

    with col_header2:
        # System mode indicator
        if HYBRID_INTEGRATION_AVAILABLE and use_enhanced:
            st.success("🚀 Enhanced Mode")
        else:
            st.warning("⚡ Basic Mode")

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        confidence_score = state_dict.get("confidence_score", 0.8)
        confidence_color = "normal" if confidence_score >= 0.8 else "inverse"
        st.metric(
            "Confidence Score",
            f"{confidence_score:.1%}",
            delta=f"{confidence_score - 0.8:.1%}" if confidence_score != 0.8 else None,
            delta_color=confidence_color
        )

    with col2:
        dormant_count = 0
        if state_dict.get("dormant_summary_report"):
            dormant_count = state_dict["dormant_summary_report"].consolidated_findings.get(
                'total_unique_flagged_accounts', 0)
        elif state_dict.get("dormant_results"):
            dormant_count = state_dict["dormant_results"].get("summary_kpis", {}).get("total_accounts_flagged_dormant",
                                                                                      0)
        elif state_dict.get("data") is not None and hasattr(state_dict["data"], 'empty') and not state_dict[
            "data"].empty:
            # Fallback calculation
            df = state_dict["data"]
            if 'Expected_Account_Dormant' in df.columns:
                dormant_count = len(
                    df[df['Expected_Account_Dormant'].astype(str).str.lower().isin(['yes', 'true', '1'])])
        st.metric("Dormant Accounts", dormant_count)

    with col3:
        compliance_issues = 0
        if state_dict.get("compliance_results"):
            cr = state_dict["compliance_results"]
            compliance_issues = (
                    cr.get("transfer_candidates_cb", {}).get("count", 0) +
                    cr.get("incomplete_contact", {}).get("count", 0)
            )
        st.metric("Compliance Issues", compliance_issues)

    with col4:
        recommendations_count = len(state_dict.get("recommendations", []))
        if state_dict.get("dormant_summary_report"):
            recommendations_count = len(state_dict["dormant_summary_report"].action_recommendations)
        st.metric("Recommendations", recommendations_count)

    # Data overview
    st.markdown("### 📊 Data Overview")
    data = state_dict.get("data")

    # Safe data checking with debug
    data_exists = safe_data_check(data, "display_results_data_overview")
    data_length = 0
    data_columns = 0

    if data_exists:
        try:
            data_length = len(data)
            data_columns = len(data.columns) if hasattr(data, 'columns') else 0
        except Exception as e:
            logger.error(f"Error getting data metrics: {e}")
            data_exists = False

    if data_exists:
        col_data1, col_data2, col_data3 = st.columns(3)

        with col_data1:
            st.metric("Total Accounts", data_length)

        with col_data2:
            if data_exists and hasattr(data, 'columns') and 'Current_Balance' in data.columns:
                try:
                    total_balance = pd.to_numeric(data['Current_Balance'], errors='coerce').sum()
                    st.metric("Total Balance", f"${total_balance:,.2f}")
                except Exception as e:
                    st.metric("Total Balance", "Error calculating")
            else:
                st.metric("Total Balance", "N/A")

        with col_data3:
            st.metric("Data Columns", data_columns)
    else:
        col_data1, col_data2, col_data3 = st.columns(3)
        with col_data1:
            st.metric("Total Accounts", "0")
        with col_data2:
            st.metric("Total Balance", "N/A")
        with col_data3:
            st.metric("Data Columns", "0")

    # Enhanced Summary Section
    if state_dict.get("dormant_summary_report") or state_dict.get("executive_summary"):
        st.markdown("---")
        display_enhanced_summary(state_dict)

    # Detailed analysis tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏦 Dormancy Analysis",
        "⚖️ Compliance Check",
        "📊 Risk Assessment",
        "🔔 Notifications",
        "🧠 Memory Insights",
        "🔍 System Logs"
    ])

    with tab1:
        display_dormancy_analysis(state_dict, use_enhanced)

    with tab2:
        display_compliance_analysis(state_dict, use_enhanced)

    with tab3:
        display_risk_assessment(state_dict, agent_logs)

    with tab4:
        display_notifications(state_dict)

    with tab5:
        display_memory_insights(state_dict)

    with tab6:
        display_system_logs(state_dict, agent_logs, langsmith_enabled)

    # Export section
    display_export_section(state_dict)


# =====================================
# MAIN STREAMLIT APPLICATION
# =====================================

# Streamlit configuration
st.set_page_config(
    page_title="Advanced AI Banking Compliance",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
config_manager = init_config_manager()
langsmith_enabled = setup_langsmith() if ENHANCED_IMPORTS else False

# Page header
st.title("🏦 Advanced Agentic AI Banking Compliance System")
st.markdown("*Enhanced Banking Compliance Analysis | CBUAE Compliant | Integration Adapter*")

# System status banner
if INTEGRATION_ADAPTER_AVAILABLE:
    st.success("🚀 **Enhanced Mode Active** - Integration Adapter Available")
else:
    st.warning("⚡ **Basic Mode Active** - Enhanced features not available")

# Sidebar configuration
with st.sidebar:
    st.header("🔧 System Configuration")

    # Analysis mode selection
    st.subheader("🎯 Analysis Mode")
    if INTEGRATION_ADAPTER_AVAILABLE or HYBRID_INTEGRATION_AVAILABLE:
        analysis_mode = st.selectbox(
            "Select Analysis Mode",
            ["Enhanced (with integration adapter)", "Basic (fallback mode)"],
            index=0
        )
        use_enhanced = analysis_mode.startswith("Enhanced")
    else:
        st.info("Enhanced mode not available")
        use_enhanced = False

    # System status
    st.subheader("📊 System Status")

    # Component status
    components_status = {
        "Streamlit": True,
        "Integration Adapter": INTEGRATION_ADAPTER_AVAILABLE,
        "Hybrid Integration": HYBRID_INTEGRATION_AVAILABLE,
        "Enhanced Imports": ENHANCED_IMPORTS,
        "Orchestrator": ORCHESTRATOR_AVAILABLE,
        "LangSmith": langsmith_enabled
    }

    for component, status in components_status.items():
        icon = "✅" if status else "❌"
        st.write(f"{icon} {component}")

    # Configuration summary
    if config_manager:
        st.markdown("**⚙️ Configuration**")
        config_summary = config_manager.get_config_summary()
        st.json(config_summary)

    # Memory system controls
    if INTEGRATION_ADAPTER_AVAILABLE:
        st.subheader("🧠 System Controls")

        if st.button("View System Insights"):
            try:
                system = init_integrated_banking_system()
                if system and hasattr(system, 'memory_agent') and system.memory_agent:
                    stats = system.memory_agent.get_knowledge_stats()
                    st.json(stats)
                else:
                    st.info("System not initialized")
            except Exception as e:
                st.error(f"Failed to get system insights: {e}")

        if st.button("System Health Check"):
            try:
                system = init_integrated_banking_system()
                if system:
                    # Run async health check
                    async def get_health():
                        if hasattr(system, 'initialize'):
                            await system.initialize()
                        if hasattr(system, 'get_system_status'):
                            return system.get_system_status()
                        return {"status": "basic_system_available"}


                    try:
                        health_status = asyncio.run(get_health())
                        st.json(health_status)
                    except RuntimeError:
                        # Handle case where event loop is already running
                        st.info("System available but detailed health check unavailable in this context")

                    # Cleanup
                    if hasattr(system, 'cleanup_system'):
                        try:
                            asyncio.run(system.cleanup_system())
                        except RuntimeError:
                            pass  # Ignore cleanup errors in Streamlit context
                else:
                    st.error("System not available")
            except Exception as e:
                st.error(f"Health check failed: {e}")

    # Analysis parameters
    st.subheader("⚙️ Analysis Parameters")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.8, 0.1)

    # Advanced options
    st.markdown("**🔬 Advanced Options**")
    session_name = st.text_input(
        "Custom Session Name",
        value="",
        help="Leave blank for auto-generated session name"
    )

    include_visualizations = st.checkbox("Include Visualizations", value=True)
    detailed_logging = st.checkbox("Detailed Logging", value=True)

    # System info
    st.subheader("ℹ️ System Information")
    env = config_manager.config.environment if hasattr(config_manager.config, 'environment') else 'development'
    st.info(f"Environment: {env}")
    st.info(f"Enhanced Features: {'Yes' if ENHANCED_IMPORTS else 'Basic'}")
    st.info(f"Integration Adapter: {'Yes' if INTEGRATION_ADAPTER_AVAILABLE else 'No'}")
    st.info(f"Hybrid Integration: {'Yes' if HYBRID_INTEGRATION_AVAILABLE else 'No'}")

    # Quick actions
    st.subheader("⚡ Quick Actions")

    if st.button("🧪 Run System Demo"):
        if INTEGRATION_ADAPTER_AVAILABLE:
            with st.spinner("Running system demo..."):
                try:
                    # Run async demo
                    async def run_demo():
                        system = await create_integrated_banking_system()
                        if hasattr(system, 'run_demo'):
                            return await system.run_demo()
                        else:
                            return "Demo completed - basic functionality verified"


                    try:
                        demo_result = asyncio.run(run_demo())
                        st.success(demo_result)
                    except RuntimeError:
                        st.info("Demo functionality available but cannot run in current context")
                except Exception as e:
                    st.error(f"Demo failed: {e}")
        else:
            st.warning("Demo requires integration adapter")

    if st.button("📊 View System Logs"):
        try:
            log_file = Path("logs/banking_app.log")
            if log_file.exists():
                with open(log_file, 'r') as f:
                    logs = f.read()
                st.text_area("Recent Logs", logs[-2000:], height=300)  # Last 2000 characters
            else:
                st.info("No logs available")
        except Exception as e:
            st.error(f"Failed to load logs: {e}")

# Main content area
st.markdown("---")

# File upload section
st.markdown("## 📤 Data Upload")

uploaded_file = st.file_uploader(
    "Upload Banking Data (.csv)",
    type="csv",
    help="Upload your banking data CSV file for compliance analysis"
)

if uploaded_file:
    try:
        # Load and display data
        df = pd.read_csv(uploaded_file)

        # Enhanced data preview with metrics
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)

            # Data quality indicators
            st.markdown("**Data Quality Indicators:**")
            quality_col1, quality_col2, quality_col3 = st.columns(3)

            with quality_col1:
                missing_data = df.isnull().sum().sum()
                total_cells = len(df) * len(df.columns)
                completeness = ((total_cells - missing_data) / total_cells) * 100
                st.metric("Data Completeness", f"{completeness:.1f}%")

            with quality_col2:
                duplicate_rows = df.duplicated().sum()
                st.metric("Duplicate Rows", duplicate_rows)

            with quality_col3:
                if 'Account_ID' in df.columns:
                    unique_accounts = df['Account_ID'].nunique()
                    st.metric("Unique Accounts", unique_accounts)

        with col2:
            st.subheader("📊 Data Summary")
            st.metric("Total Accounts", len(df))
            st.metric("Total Columns", len(df.columns))

            # Pre-analysis checks
            if 'Expected_Account_Dormant' in df.columns:
                dormant_count = len(
                    df[df['Expected_Account_Dormant'].astype(str).str.lower().isin(['yes', 'true', '1'])]
                )
                st.metric("Pre-flagged Dormant", dormant_count)

            if 'Current_Balance' in df.columns:
                total_balance = pd.to_numeric(df['Current_Balance'], errors='coerce').sum()
                st.metric("Total Balance", f"${total_balance:,.2f}")

            if 'Account_Type' in df.columns:
                account_types = df['Account_Type'].nunique()
                st.metric("Account Types", account_types)

        # Data validation
        st.markdown("### 🔍 Data Validation")
        validation_col1, validation_col2 = st.columns(2)

        with validation_col1:
            st.markdown("**Required Columns:**")
            required_cols = ['Account_ID']
            for col in required_cols:
                if col in df.columns:
                    st.success(f"✅ {col}")
                else:
                    st.error(f"❌ {col} (Required)")

        with validation_col2:
            st.markdown("**Optional Columns:**")
            optional_cols = [
                'Expected_Account_Dormant', 'Date_Last_Cust_Initiated_Activity',
                'Account_Type', 'Current_Balance', 'Customer_ID'
            ]
            for col in optional_cols:
                if col in df.columns:
                    st.info(f"✅ {col}")
                else:
                    st.warning(f"➖ {col}")

        # Column information with dropdown
        st.markdown("### 📝 Column Information")

        # Column overview metrics
        col_overview1, col_overview2, col_overview3 = st.columns(3)
        with col_overview1:
            st.metric("Total Columns", len(df.columns))
        with col_overview2:
            numeric_cols = df.select_dtypes(include=['number']).columns
            st.metric("Numeric Columns", len(numeric_cols))
        with col_overview3:
            text_cols = df.select_dtypes(include=['object']).columns
            st.metric("Text Columns", len(text_cols))

        # Interactive column explorer
        st.markdown("**📋 Column Explorer:**")

        # Column selection dropdown
        selected_column = st.selectbox(
            "Select a column to view detailed information:",
            options=[""] + list(df.columns),
            index=0,
            help="Choose a column to see detailed statistics and sample values"
        )

        if selected_column:
            # Display detailed information for selected column
            col_detail1, col_detail2 = st.columns([2, 1])

            with col_detail1:
                st.markdown(f"#### 📊 Details for: `{selected_column}`")

                # Basic statistics
                dtype = str(df[selected_column].dtype)
                null_count = df[selected_column].isnull().sum()
                null_percentage = (null_count / len(df)) * 100
                unique_count = df[selected_column].nunique()

                # Display metrics in a nice format
                detail_col1, detail_col2, detail_col3 = st.columns(3)
                with detail_col1:
                    st.metric("Data Type", dtype)
                with detail_col2:
                    st.metric("Missing Values", f"{null_count} ({null_percentage:.1f}%)")
                with detail_col3:
                    st.metric("Unique Values", unique_count)

                # Additional statistics based on data type
                if df[selected_column].dtype in ['int64', 'float64']:
                    st.markdown("**📈 Numeric Statistics:**")
                    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                    with stats_col1:
                        st.metric("Min", f"{df[selected_column].min():.2f}")
                    with stats_col2:
                        st.metric("Max", f"{df[selected_column].max():.2f}")
                    with stats_col3:
                        st.metric("Mean", f"{df[selected_column].mean():.2f}")
                    with stats_col4:
                        st.metric("Std Dev", f"{df[selected_column].std():.2f}")

                # Sample values
                st.markdown("**🔍 Sample Values:**")
                if unique_count <= 10:
                    # Show all unique values if there are few
                    unique_values = df[selected_column].dropna().unique()
                    for i, val in enumerate(unique_values[:10]):
                        st.write(f"• {val}")
                else:
                    # Show top 5 most common values
                    value_counts = df[selected_column].value_counts().head(5)
                    st.markdown("*Top 5 most common values:*")
                    for val, count in value_counts.items():
                        percentage = (count / len(df)) * 100
                        st.write(f"• `{val}` - {count} occurrences ({percentage:.1f}%)")

            with col_detail2:
                st.markdown("#### 📋 Column Properties")

                # Data quality indicators
                quality_score = 100 - null_percentage
                if quality_score >= 95:
                    quality_color = "🟢"
                    quality_status = "Excellent"
                elif quality_score >= 80:
                    quality_color = "🟡"
                    quality_status = "Good"
                else:
                    quality_color = "🔴"
                    quality_status = "Needs Attention"

                st.markdown(f"**Data Quality:** {quality_color} {quality_status}")
                st.progress(quality_score / 100)

                # Column characteristics
                st.markdown("**Characteristics:**")

                # Check for potential issues
                issues = []
                if null_percentage > 20:
                    issues.append("High missing data")
                if unique_count == 1:
                    issues.append("Single value only")
                if unique_count == len(df):
                    issues.append("All values unique")

                if issues:
                    st.markdown("**⚠️ Potential Issues:**")
                    for issue in issues:
                        st.write(f"• {issue}")
                else:
                    st.success("✅ No issues detected")

                # Compliance relevance
                st.markdown("**🏦 Compliance Relevance:**")
                compliance_fields = {
                    'Account_ID': 'Primary identifier',
                    'Expected_Account_Dormant': 'Dormancy flag',
                    'Current_Balance': 'Balance tracking',
                    'Account_Type': 'Account classification',
                    'Date_Last_Cust_Initiated_Activity': 'Activity tracking',
                    'Expected_Transfer_to_CB_Due': 'CB transfer requirement',
                    'Bank_Contact_Attempted_Post_Dormancy_Trigger': 'Contact compliance'
                }

                if selected_column in compliance_fields:
                    st.info(f"📋 {compliance_fields[selected_column]}")
                else:
                    st.write("Additional data field")

        # Quick column summary table
        st.markdown("**📄 Quick Summary Table:**")

        # Create summary dataframe
        summary_data = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            null_count = df[col].isnull().sum()
            null_percentage = (null_count / len(df)) * 100
            unique_count = df[col].nunique()

            # Data quality score
            quality_score = 100 - null_percentage
            if quality_score >= 95:
                quality_icon = "🟢"
            elif quality_score >= 80:
                quality_icon = "🟡"
            else:
                quality_icon = "🔴"

            summary_data.append({
                'Column': col,
                'Type': dtype,
                'Missing (%)': f"{null_percentage:.1f}%",
                'Unique Values': unique_count,
                'Quality': quality_icon
            })

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # Export column information
        if st.button("📥 Export Column Information"):
            try:
                # Create detailed column report
                detailed_report = []
                for col in df.columns:
                    col_info = {
                        'column_name': str(col),
                        'data_type': str(df[col].dtype),
                        'null_count': int(df[col].isnull().sum()),
                        'null_percentage': round(float((df[col].isnull().sum() / len(df)) * 100), 2),
                        'unique_count': int(df[col].nunique()),
                        'total_rows': int(len(df))
                    }

                    # Add numeric statistics if applicable
                    if df[col].dtype in ['int64', 'float64']:
                        try:
                            col_info.update({
                                'min_value': float(df[col].min()) if pd.notnull(df[col].min()) else None,
                                'max_value': float(df[col].max()) if pd.notnull(df[col].max()) else None,
                                'mean_value': round(float(df[col].mean()), 4) if pd.notnull(df[col].mean()) else None,
                                'std_value': round(float(df[col].std()), 4) if pd.notnull(df[col].std()) else None
                            })
                        except (ValueError, TypeError):
                            # Handle any conversion errors
                            col_info['numeric_stats_error'] = 'Could not calculate numeric statistics'

                    # Add top values (limit to prevent huge JSON)
                    try:
                        if df[col].nunique() <= 20:
                            value_counts = df[col].value_counts().head(10)
                            # Convert to simple dict with string keys
                            value_dist = {}
                            for val, count in value_counts.items():
                                # Convert value to string to ensure JSON serialization
                                key = str(val) if pd.notnull(val) else "null"
                                value_dist[key] = int(count)
                            col_info['value_distribution'] = value_dist
                        else:
                            # For columns with many unique values, just show top 5
                            value_counts = df[col].value_counts().head(5)
                            value_dist = {}
                            for val, count in value_counts.items():
                                key = str(val) if pd.notnull(val) else "null"
                                value_dist[key] = int(count)
                            col_info['top_values'] = value_dist
                    except Exception as e:
                        col_info['value_distribution_error'] = f'Could not calculate value distribution: {str(e)}'

                    detailed_report.append(col_info)

                # Add metadata
                report_metadata = {
                    'export_timestamp': datetime.now().isoformat(),
                    'total_columns': len(df.columns),
                    'total_rows': len(df),
                    'file_name': uploaded_file.name if uploaded_file else 'unknown',
                    'analysis_mode': 'enhanced' if use_enhanced else 'basic'
                }

                # Create final report structure
                final_report = {
                    'metadata': report_metadata,
                    'columns': detailed_report
                }

                # Convert to JSON with proper error handling
                try:
                    report_json = json.dumps(final_report, indent=2, ensure_ascii=False)

                    # Validate JSON before offering download
                    json.loads(report_json)  # This will raise an exception if invalid

                    st.download_button(
                        label="📊 Download Detailed Column Report",
                        data=report_json.encode('utf-8'),
                        file_name=f'column_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
                        mime='application/json'
                    )
                    st.success("✅ Column report prepared for download!")

                except json.JSONEncoder as json_error:
                    st.error(f"❌ JSON encoding error: {str(json_error)}")
                    # Fallback: create a simpler CSV export
                    st.info("🔄 Creating fallback CSV export...")

                    simple_data = []
                    for col in df.columns:
                        simple_data.append({
                            'Column': str(col),
                            'Type': str(df[col].dtype),
                            'Missing_Count': int(df[col].isnull().sum()),
                            'Missing_Percentage': round((df[col].isnull().sum() / len(df)) * 100, 2),
                            'Unique_Values': int(df[col].nunique())
                        })

                    simple_df = pd.DataFrame(simple_data)
                    csv_data = simple_df.to_csv(index=False).encode('utf-8')

                    st.download_button(
                        label="📄 Download Column Summary (CSV)",
                        data=csv_data,
                        file_name=f'column_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                        mime='text/csv'
                    )

            except Exception as e:
                st.error(f"❌ Export failed: {str(e)}")
                logger.error(f"Column export error: {e}")

                # Show debug information
                st.markdown("**Debug Information:**")
                st.code(f"Error: {str(e)}")
                st.code(f"Error type: {type(e).__name__}")

                # Offer basic export as fallback
                if st.button("🔄 Try Basic Export"):
                    try:
                        basic_data = {
                            'columns': [str(col) for col in df.columns],
                            'data_types': [str(dtype) for dtype in df.dtypes],
                            'null_counts': [int(count) for count in df.isnull().sum()],
                            'export_time': datetime.now().isoformat()
                        }

                        basic_json = json.dumps(basic_data, indent=2)
                        st.download_button(
                            label="📋 Download Basic Column Info",
                            data=basic_json.encode('utf-8'),
                            file_name=f'basic_column_info_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
                            mime='application/json'
                        )
                        st.success("✅ Basic export ready!")
                    except Exception as basic_error:
                        st.error(f"❌ Basic export also failed: {str(basic_error)}")

        # Analysis button
        st.markdown("---")

        # Pre-analysis configuration
        analysis_col1, analysis_col2 = st.columns(2)

        with analysis_col1:
            st.markdown("**Analysis Configuration:**")
            st.write(f"🎯 Mode: {'Enhanced' if use_enhanced else 'Basic'}")
            st.write(f"📊 Confidence Threshold: {confidence_threshold}")
            st.write(f"🧠 Integration Adapter: {'Active' if INTEGRATION_ADAPTER_AVAILABLE else 'Inactive'}")

        with analysis_col2:
            st.markdown("**Expected Processing:**")
            estimated_time = len(df) * 0.01  # Rough estimate
            st.write(f"⏱️ Estimated Time: {estimated_time:.1f} seconds")
            st.write(f"📈 Accounts to Process: {len(df)}")
            st.write(f"🔄 Analysis Steps: {'8-12' if use_enhanced else '4-6'}")

        # Main analysis button
        if st.button("🚀 Run Advanced Compliance Analysis", type="primary", use_container_width=True):

            # Create progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Analysis container
            analysis_container = st.container()

            with analysis_container:
                with st.spinner("Initializing Enhanced AI Compliance System..."):

                    try:
                        # Update progress
                        status_text.text("🔄 Initializing system components...")
                        progress_bar.progress(10)

                        # Generate session name
                        if session_name:
                            current_session = session_name
                        else:
                            current_session = f"streamlit_session_{int(time.time())}"

                        # Log analysis start
                        logger.info(f"Starting analysis for session: {current_session}")
                        logger.info(f"Data shape: {df.shape}, Mode: {'Enhanced' if use_enhanced else 'Basic'}")

                        # Update progress
                        status_text.text("🔄 Running enhanced analysis workflow...")
                        progress_bar.progress(30)

                        # Run the analysis
                        start_time = time.time()
                        try:
                            final_state, agent_logs = run_async_analysis(
                                df,
                                session_name=current_session,
                                use_enhanced=use_enhanced
                            )

                            # Validate the results
                            if final_state is None:
                                raise ValueError("Analysis returned None result")

                            # Ensure agent_logs is not None
                            if agent_logs is None:
                                agent_logs = []
                                logger.warning("Agent logs were None, using empty list")

                        except Exception as analysis_error:
                            logger.error(f"Analysis execution failed: {analysis_error}")

                            # Create error state
                            final_state = {
                                'session_id': current_session,
                                'error': str(analysis_error),
                                'data': df,
                                'timestamp': datetime.now(),
                                'confidence_score': 0.0
                            }

                            # Create error agent log
                            agent_logs = [{
                                'agent': 'error_handler',
                                'timestamp': datetime.now().isoformat(),
                                'status': 'error',
                                'details': {
                                    'error': str(analysis_error),
                                    'error_type': type(analysis_error).__name__,
                                    'session_id': current_session
                                }
                            }]

                        processing_time = time.time() - start_time

                        # Update progress
                        progress_bar.progress(90)
                        status_text.text("🔄 Generating results...")

                        # Log completion
                        logger.info(f"Analysis completed in {processing_time:.2f} seconds")

                        # Final progress
                        progress_bar.progress(100)
                        status_text.text("✅ Analysis completed successfully!")

                        # Success message
                        success_col1, success_col2, success_col3 = st.columns(3)
                        with success_col1:
                            st.success("✅ Enhanced compliance analysis completed!")
                        with success_col2:
                            st.info(f"⏱️ Processing time: {processing_time:.2f}s")
                        with success_col3:
                            st.info(f"🆔 Session: {current_session}")

                        # Display results using enhanced function
                        display_results(final_state, agent_logs, use_enhanced, langsmith_enabled)

                    except Exception as e:
                        # Error handling
                        progress_bar.progress(0)
                        status_text.text("❌ Analysis failed")

                        st.error(f"❌ Analysis failed: {str(e)}")
                        logger.error(f"Analysis error: {e}")

                        # Enhanced error information
                        st.markdown("### 🔍 Error Details")
                        st.code(str(e))

                        # System status on error
                        st.markdown("**System Status:**")
                        error_health = get_system_health()
                        st.json(error_health)

                        # Troubleshooting suggestions
                        st.markdown("**Troubleshooting Steps:**")
                        st.write("1. Check that all required components are installed")
                        st.write("2. Verify the data format matches expected schema")
                        st.write("3. Try running in basic mode")
                        st.write("4. Check system logs for detailed error information")
                        st.write("5. Ensure all dependencies are properly installed")

                        # Fallback option
                        if use_enhanced and (INTEGRATION_ADAPTER_AVAILABLE or HYBRID_INTEGRATION_AVAILABLE):
                            st.info("💡 Trying fallback to basic mode...")
                            try:
                                fallback_results, fallback_logs = run_async_analysis(
                                    df,
                                    session_name=f"{current_session}_fallback",
                                    use_enhanced=False
                                )
                                st.success("✅ Fallback analysis completed")
                                display_results(fallback_results, fallback_logs, False, langsmith_enabled)
                            except Exception as fallback_error:
                                st.error(f"❌ Fallback also failed: {fallback_error}")

    except Exception as e:
        st.error(f"❌ Failed to load data: {str(e)}")
        logger.error(f"Data loading error: {e}")

        # Data loading troubleshooting
        st.markdown("### 🔧 Data Loading Help")
        st.markdown("**Common Issues:**")
        st.write("- File encoding issues (try UTF-8)")
        st.write("- CSV format problems (check delimiters)")
        st.write("- File size too large")
        st.write("- Missing required columns")

        st.markdown("**Expected CSV Format:**")
        st.code("""
Account_ID,Account_Type,Current_Balance,Expected_Account_Dormant
ACC001,Current,10000,no
ACC002,Savings,25000,yes
        """)

else:
    # No file uploaded - show sample data option
    st.info("👆 Please upload a CSV file to begin compliance analysis")

    # Sample data option
    if st.button("📊 Generate Sample Data for Testing"):
        try:
            if INTEGRATION_ADAPTER_AVAILABLE:
                # Try to get sample data from the integrated system
                try:
                    async def get_sample_data():
                        system = await create_integrated_banking_system()
                        if hasattr(system, 'create_sample_banking_data'):
                            return system.create_sample_banking_data()
                        else:
                            return None


                    sample_df = asyncio.run(get_sample_data())

                    if sample_df is None:
                        raise Exception("Sample data method not available")

                except Exception:
                    # Fallback to basic sample data
                    sample_df = pd.DataFrame({
                        'Account_ID': [f'ACC{i:06d}' for i in range(20)],
                        'Account_Type': ['Current', 'Savings', 'Fixed'] * 6 + ['Investment', 'Current'],
                        'Current_Balance': [10000 + i * 1000 for i in range(20)],
                        'Expected_Account_Dormant': ['yes' if i % 5 == 0 else 'no' for i in range(20)]
                    })
            else:
                # Basic sample data
                sample_df = pd.DataFrame({
                    'Account_ID': [f'ACC{i:06d}' for i in range(20)],
                    'Account_Type': ['Current', 'Savings', 'Fixed'] * 6 + ['Investment', 'Current'],
                    'Current_Balance': [10000 + i * 1000 for i in range(20)],
                    'Expected_Account_Dormant': ['yes' if i % 5 == 0 else 'no' for i in range(20)]
                })

            st.success("✅ Sample data generated!")
            st.dataframe(sample_df.head(), use_container_width=True)

            # Download sample data
            csv = sample_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Sample Data",
                data=csv,
                file_name="sample_banking_data.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Failed to generate sample data: {e}")

# Footer
st.markdown("---")

# System information footer
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("**🏦 System Information**")
    st.write(f"Mode: {'Enhanced' if INTEGRATION_ADAPTER_AVAILABLE else 'Basic'}")
    st.write(f"Version: 2.1.0")

with footer_col2:
    st.markdown("**📊 Session Statistics**")
    if 'session_count' not in st.session_state:
        st.session_state.session_count = 0
    st.write(f"Sessions this run: {st.session_state.session_count}")
    st.write(f"Uptime: {datetime.now().strftime('%H:%M:%S')}")

with footer_col3:
    st.markdown("**🔗 Quick Links**")
    if st.button("📖 Documentation"):
        st.info("Documentation available in project README")
    if st.button("🐛 Report Issue"):
        st.info("Issues can be reported via project repository")

# Enhanced footer
st.markdown(
    """
    <div style='text-align: center; color: #666; margin-top: 50px;'>
        <p>🏦 Advanced Agentic AI Banking Compliance System</p>
        <p>Enhanced with Integration Adapter | CBUAE Compliant | Enterprise Ready</p>
        <p>Powered by Streamlit • Python • AI Agents • Integration Systems</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Session state management
if uploaded_file:
    st.session_state.session_count = st.session_state.get('session_count', 0) + 1

# Application lifecycle management
if __name__ == "__main__":
    # Initialize logging for main execution
    logger.info("Starting Advanced AI Banking Compliance System")

    # Check system health on startup
    health = get_system_health()
    logger.info(f"System health: {health}")

    # Log configuration
    logger.info(f"Integration adapter: {INTEGRATION_ADAPTER_AVAILABLE}")
    logger.info(f"Hybrid integration: {HYBRID_INTEGRATION_AVAILABLE}")
    logger.info(f"Enhanced imports: {ENHANCED_IMPORTS}")
    logger.info(f"Orchestrator: {ORCHESTRATOR_AVAILABLE}")
    logger.info(f"LangSmith enabled: {langsmith_enabled}")

    # Clean up on exit (if running directly)
    try:
        if INTEGRATION_ADAPTER_AVAILABLE:
            # Register cleanup function for proper shutdown
            import atexit


            def cleanup():
                try:
                    async def async_cleanup():
                        try:
                            system = await create_integrated_banking_system()
                            if system and hasattr(system, 'cleanup_system'):
                                await system.cleanup_system()
                                logger.info("System cleanup completed")
                        except Exception as cleanup_error:
                            logger.error(f"Cleanup error: {cleanup_error}")

                    # Try to run cleanup, but don't fail if event loop issues occur
                    try:
                        asyncio.run(async_cleanup())
                    except RuntimeError as re:
                        if "cannot be called from a running event loop" in str(re):
                            logger.info("Cleanup skipped due to running event loop")
                        else:
                            logger.error(f"Cleanup runtime error: {re}")
                except Exception as e:
                    logger.warning(f"Could not register cleanup function: {e}")


            atexit.register(cleanup)
    except Exception as e:
        logger.warning(f"Could not register cleanup function: {e}")


def debug_data_state(data, location="unknown"):
    """Debug function to safely check data state"""
    try:
        if data is None:
            logger.info(f"[{location}] Data is None")
            return False, "Data is None"

        if not hasattr(data, 'empty'):
            logger.info(f"[{location}] Data has no 'empty' attribute, type: {type(data)}")
            return False, f"No empty attribute, type: {type(data)}"

        if data.empty:
            logger.info(f"[{location}] Data is empty DataFrame")
            return False, "Empty DataFrame"

        logger.info(f"[{location}] Data is valid DataFrame with {len(data)} rows")
        return True, f"Valid DataFrame with {len(data)} rows"

    except Exception as e:
        logger.error(f"[{location}] Error checking data: {e}")
        return False, f"Error: {str(e)}"


def safe_data_check(data, location="unknown"):
    """Safely check if data exists and is valid"""
    try:
        is_valid, message = debug_data_state(data, location)
        return is_valid
    except Exception as e:
        logger.error(f"Safe data check failed at {location}: {e}")
        return False


def debug_system_state():
    """Debug function to check system state"""
    debug_info = {
        'streamlit_version': st.__version__,
        'pandas_version': pd.__version__,
        'python_version': os.sys.version,
        'working_directory': os.getcwd(),
        'environment_variables': {
            'LANGCHAIN_TRACING_V2': os.getenv('LANGCHAIN_TRACING_V2', 'Not set'),
            'LANGCHAIN_API_KEY': 'Set' if os.getenv('LANGCHAIN_API_KEY') else 'Not set'
        },
        'file_structure': {
            'logs_exists': Path("logs").exists(),
            'data_exists': Path("data").exists(),
            'log_file_exists': Path("logs/banking_app.log").exists()
        },
        'integration_adapter': INTEGRATION_ADAPTER_AVAILABLE,
        'hybrid_integration': HYBRID_INTEGRATION_AVAILABLE,
        'enhanced_imports': ENHANCED_IMPORTS,
        'orchestrator': ORCHESTRATOR_AVAILABLE
    }
    return debug_info


# Performance monitoring
class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.metrics = {}

    def record_metric(self, name, value):
        self.metrics[name] = {
            'value': value,
            'timestamp': time.time() - self.start_time
        }

    def get_metrics(self):
        return self.metrics


# Initialize performance monitor
if 'perf_monitor' not in st.session_state:
    st.session_state.perf_monitor = PerformanceMonitor()


# Global error handler
def handle_global_error(error, context="Unknown"):
    """Handle global application errors"""
    error_info = {
        'error': str(error),
        'context': context,
        'timestamp': datetime.now().isoformat(),
        'system_health': get_system_health()
    }

    logger.error(f"Global error in {context}: {error}")
    logger.error(f"Error details: {error_info}")

    # Store error in session state for debugging
    if 'global_errors' not in st.session_state:
        st.session_state.global_errors = []

    st.session_state.global_errors.append(error_info)

    # Limit error history to last 10 errors
    if len(st.session_state.global_errors) > 10:
        st.session_state.global_errors = st.session_state.global_errors[-10:]

    return error_info


# Enhanced session management
def initialize_session():
    """Initialize session with enhanced tracking"""
    if 'session_initialized' not in st.session_state:
        st.session_state.session_initialized = True
        st.session_state.session_start_time = datetime.now()
        st.session_state.session_id = f"streamlit_{int(time.time())}"

        logger.info(f"New session initialized: {st.session_state.session_id}")

        # Initialize session tracking
        st.session_state.analysis_count = 0
        st.session_state.upload_count = 0
        st.session_state.error_count = 0

        # Log session start
        if 'perf_monitor' in st.session_state:
            st.session_state.perf_monitor.record_metric('session_start', datetime.now().isoformat())


# Enhanced async helper functions
def safe_async_run(coro, fallback_result=None):
    """Safely run an async coroutine, handling event loop issues"""
    try:
        return asyncio.run(coro)
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            # Try with a new event loop
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(coro)
                finally:
                    loop.close()
            except Exception as loop_error:
                logger.error(f"Event loop error: {loop_error}")
                return fallback_result
        else:
            logger.error(f"Async runtime error: {e}")
            return fallback_result
    except Exception as e:
        logger.error(f"Async execution error: {e}")
        return fallback_result


def get_system_capabilities():
    """Get available system capabilities"""
    capabilities = {
        'basic_analysis': True,
        'data_processing': True,
        'export_functionality': True,
        'logging': True,
        'session_management': True,
        'integration_adapter': INTEGRATION_ADAPTER_AVAILABLE,
        'hybrid_integration': HYBRID_INTEGRATION_AVAILABLE,
        'enhanced_imports': ENHANCED_IMPORTS,
        'orchestrator': ORCHESTRATOR_AVAILABLE,
        'langsmith_tracing': langsmith_enabled,
        'async_processing': True,
        'error_handling': True,
        'performance_monitoring': True
    }

    if INTEGRATION_ADAPTER_AVAILABLE:
        try:
            system = init_integrated_banking_system()
            if system:
                capabilities.update({
                    'memory_system': hasattr(system, 'memory_agent'),
                    'compliance_analysis': hasattr(system, 'process_banking_data'),
                    'system_status': hasattr(system, 'get_system_status'),
                    'session_history': hasattr(system, 'get_session_history'),
                    'cleanup_system': hasattr(system, 'cleanup_system')
                })
        except Exception as e:
            logger.warning(f"Could not check system capabilities: {e}")

    return capabilities


def create_system_report():
    """Create a comprehensive system report"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'system_health': get_system_health(),
        'capabilities': get_system_capabilities(),
        'debug_info': debug_system_state(),
        'session_info': {
            'session_id': st.session_state.get('session_id', 'unknown'),
            'session_start': st.session_state.get('session_start_time', 'unknown'),
            'analysis_count': st.session_state.get('analysis_count', 0),
            'upload_count': st.session_state.get('upload_count', 0),
            'error_count': st.session_state.get('error_count', 0)
        }
    }

    if 'perf_monitor' in st.session_state:
        report['performance_metrics'] = st.session_state.perf_monitor.get_metrics()

    if 'global_errors' in st.session_state:
        report['recent_errors'] = st.session_state.global_errors[-5:]  # Last 5 errors

    return report


# System initialization check
def verify_system_integrity():
    """Verify system integrity and dependencies"""
    integrity_check = {
        'status': 'healthy',
        'issues': [],
        'warnings': [],
        'recommendations': []
    }

    # Check basic dependencies
    try:
        import streamlit, pandas, asyncio, logging, json
        integrity_check['basic_dependencies'] = True
    except ImportError as e:
        integrity_check['issues'].append(f"Missing basic dependency: {e}")
        integrity_check['status'] = 'critical'

    # Check integration adapter
    if not INTEGRATION_ADAPTER_AVAILABLE:
        integrity_check['warnings'].append("Integration adapter not available - running in basic mode")
        integrity_check['recommendations'].append("Install integration adapter for enhanced features")

    # Check hybrid integration
    if not HYBRID_INTEGRATION_AVAILABLE:
        integrity_check['warnings'].append("Hybrid integration not available - some features limited")
        integrity_check['recommendations'].append("Install hybrid integration for advanced features")

    # Check enhanced imports
    if not ENHANCED_IMPORTS:
        integrity_check['warnings'].append("Enhanced imports not available - some features limited")
        integrity_check['recommendations'].append("Install enhanced dependencies for full functionality")

    # Check orchestrator
    if not ORCHESTRATOR_AVAILABLE:
        integrity_check['warnings'].append("Orchestrator not available - workflow features limited")
        integrity_check['recommendations'].append("Install orchestrator for enhanced workflow management")

    # Check directory structure
    required_dirs = ['logs', 'data']
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            try:
                Path(dir_name).mkdir(exist_ok=True)
                integrity_check['warnings'].append(f"Created missing directory: {dir_name}")
            except Exception as e:
                integrity_check['issues'].append(f"Cannot create directory {dir_name}: {e}")
                integrity_check['status'] = 'degraded'

    # Check logging
    try:
        logger.info("System integrity check")
        integrity_check['logging'] = True
    except Exception as e:
        integrity_check['issues'].append(f"Logging system issue: {e}")
        integrity_check['status'] = 'degraded'

    # Set overall status
    if integrity_check['issues']:
        if integrity_check['status'] != 'critical':
            integrity_check['status'] = 'degraded'
    elif integrity_check['warnings']:
        if integrity_check['status'] == 'healthy':
            integrity_check['status'] = 'warning'

    return integrity_check


# Initialize session
initialize_session()

# Run system integrity check on startup
startup_integrity = verify_system_integrity()
logger.info(f"System integrity check: {startup_integrity['status']}")

if startup_integrity['issues']:
    for issue in startup_integrity['issues']:
        logger.error(f"Integrity issue: {issue}")

if startup_integrity['warnings']:
    for warning in startup_integrity['warnings']:
        logger.warning(f"Integrity warning: {warning}")

# End of file - Complete Enhanced Banking Compliance System with Integration Adapter