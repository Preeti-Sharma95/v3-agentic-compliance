# main.py - Fixed Structure
import streamlit as st
import pandas as pd
import os
from datetime import datetime
import logging

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Import enhanced orchestrator
from enhanced_orchestrator import EnhancedComplianceOrchestrator

# Import configuration and utilities
try:
    from config.config import config
    from utils.langsmith_setup import setup_langsmith, get_trace_url
    from utils.visualization_helpers import (
        create_dormancy_breakdown_chart,
        create_compliance_scatter_chart,
        create_risk_gauge,
        create_notification_pie_chart
    )

    ENHANCED_IMPORTS = True
except ImportError:
    ENHANCED_IMPORTS = False
    # Fallback setup
    config = type('Config', (), {
        'langsmith_enabled': False,
        'mcp_enabled': False,
        'environment': 'development'
    })()


    def setup_langsmith():
        return False


    def get_trace_url(session_id):
        return None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =====================================
# FUNCTION DEFINITIONS (MUST BE FIRST)
# =====================================

def display_results(final_state, agent_logs, use_enhanced, langsmith_enabled):
    """Display analysis results"""

    # Handle both dict and object states
    if hasattr(final_state, '__dict__'):
        state_dict = final_state.__dict__
    else:
        state_dict = final_state

    # Error handling
    if state_dict.get("error"):
        st.error(f"❌ Analysis Error: {state_dict['error']}")
        return

    # Main results summary
    st.markdown("## 📈 Analysis Results")

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        confidence_score = state_dict.get("confidence_score", 0.8)
        confidence_color = "normal" if confidence_score >= 0.8 else "inverse"
        st.metric(
            "Confidence Score",
            f"{confidence_score:.1%}",
            delta=f"{confidence_score - 0.8:.1%}",
            delta_color=confidence_color
        )

    with col2:
        dormant_count = 0
        if state_dict.get("dormant_results"):
            dormant_count = state_dict["dormant_results"].get("summary_kpis", {}).get("total_accounts_flagged_dormant",
                                                                                      0)
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
        st.metric("Recommendations", recommendations_count)

    # Detailed analysis tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏦 Dormancy Analysis",
        "⚖️ Compliance Check",
        "📊 Risk Assessment",
        "🔔 Notifications",
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
        display_system_logs(state_dict, agent_logs, langsmith_enabled)

    # Export section
    display_export_section(state_dict)


def display_dormancy_analysis(state_dict, use_enhanced):
    """Display dormancy analysis results"""
    if state_dict.get("dormant_results"):
        dr = state_dict["dormant_results"]
        kpis = dr.get("summary_kpis", {})

        # Dormancy overview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Dormancy Rate", f"{kpis.get('percentage_dormant_of_total', 0):.1f}%")
        with col2:
            st.metric("High Value Dormant", kpis.get('count_high_value_dormant', 0))
        with col3:
            st.metric("CB Transfer Eligible", kpis.get('count_eligible_for_cb_transfer', 0))

        # Simple dormancy breakdown
        breakdown_data = {
            'Safe Deposit': kpis.get('count_sdb_dormant', 0),
            'Investment': kpis.get('count_investment_dormant', 0),
            'Fixed Deposit': kpis.get('count_fixed_deposit_dormant', 0),
            'Demand Deposit': kpis.get('count_demand_deposit_dormant', 0),
            'Unclaimed Instruments': kpis.get('count_unclaimed_instruments', 0)
        }

        total_dormant = sum(breakdown_data.values())
        if total_dormant > 0:
            st.markdown("### Dormant Accounts Breakdown")
            for category, count in breakdown_data.items():
                if count > 0:
                    st.metric(category, count)
        else:
            st.info("No dormant accounts found in detailed categories.")

        # Enhanced features placeholder
        if use_enhanced:
            with st.expander("📚 Enhanced Analysis"):
                st.info("🔗 Enhanced regulatory guidance and MCP features available")
    else:
        st.info("No dormancy analysis results available.")


def display_compliance_analysis(state_dict, use_enhanced):
    """Display compliance analysis results"""
    if state_dict.get("compliance_results"):
        cr = state_dict["compliance_results"]

        # Critical compliance metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            count = cr.get('transfer_candidates_cb', {}).get('count', 0)
            st.metric("🚨 CB Transfer Due", count)
            if count > 0:
                st.error("Immediate action required!")

        with col2:
            count = cr.get('incomplete_contact', {}).get('count', 0)
            st.metric("📞 Incomplete Contact", count)

        with col3:
            count = cr.get('flag_candidates', {}).get('count', 0)
            st.metric("🏷️ Unflagged Dormant", count)

        # Additional compliance metrics
        additional_metrics = {
            'Internal Ledger Transfer': cr.get('ledger_candidates_internal', {}).get('count', 0),
            'Statement Freeze Needed': cr.get('statement_freeze_needed', {}).get('count', 0)
        }

        st.markdown("### Additional Compliance Checks")
        for metric, value in additional_metrics.items():
            if value > 0:
                st.warning(f"**{metric}:** {value} accounts")
            else:
                st.success(f"**{metric}:** No issues found")

        # Enhanced features placeholder
        if use_enhanced:
            with st.expander("🔍 Enhanced Validation"):
                st.info("🔗 MCP validation and enhanced compliance features available")
    else:
        st.info("No compliance analysis results available.")


def display_risk_assessment(state_dict, agent_logs):
    """Display risk assessment results"""
    # Risk recommendations
    recommendations = state_dict.get("recommendations", [])
    if recommendations:
        st.markdown("### 🎯 Risk-Based Recommendations")
        for i, rec in enumerate(recommendations, 1):
            priority = "🔴" if "URGENT" in rec else "🟡" if "WARNING" in rec else "🟢"
            st.markdown(f"{priority} **{i}.** {rec}")

    # Confidence score display
    confidence_score = state_dict.get("confidence_score", 0.8)
    st.markdown("### 📊 Confidence Assessment")

    # Simple confidence display
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Overall Confidence", f"{confidence_score:.1%}")
    with col2:
        risk_level = "Low" if confidence_score >= 0.8 else "Medium" if confidence_score >= 0.6 else "High"
        color = "green" if risk_level == "Low" else "orange" if risk_level == "Medium" else "red"
        st.markdown(f"**Risk Level:** :{color}[{risk_level}]")

    # Progress bar for confidence
    st.progress(confidence_score)


def display_notifications(state_dict):
    """Display system notifications"""
    st.markdown("### 🔔 System Notifications")

    notifications = state_dict.get("notifications", [])
    if notifications:
        for notification in notifications:
            if "URGENT" in notification:
                st.error(notification)
            elif "WARNING" in notification:
                st.warning(notification)
            else:
                st.info(notification)
    else:
        st.success("✅ No critical notifications at this time.")


def display_system_logs(state_dict, agent_logs, langsmith_enabled):
    """Display system logs and traces"""
    st.markdown("### 🔍 System Execution Logs")

    # LangSmith trace link
    if langsmith_enabled and state_dict.get("session_id"):
        trace_url = get_trace_url(state_dict["session_id"])
        if trace_url:
            st.markdown(f"🔗 [View in LangSmith]({trace_url})")
        st.markdown(f"**Session ID:** `{state_dict['session_id']}`")

    # Agent performance
    if agent_logs:
        st.markdown("#### Execution Summary")
        success_count = sum(1 for log in agent_logs if log.get("status") == "success")
        total_count = len(agent_logs)
        st.metric("Success Rate", f"{success_count}/{total_count}")

        # Detailed logs
        with st.expander("📝 Detailed Execution Logs"):
            for i, log in enumerate(agent_logs):
                status_icon = "✅" if log.get("status") == "success" else "❌"
                agent_name = log.get("agent", "unknown").replace("_", " ").title()
                timestamp = log.get("timestamp", "unknown")

                st.markdown(f"**{status_icon} {agent_name}** - {timestamp}")

                if log.get("details"):
                    st.json(log["details"])
                st.divider()
    else:
        st.info("No execution logs available.")


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
            'notifications': state_dict.get("notifications", [])
        }

        report_json = pd.Series(report_data).to_json(indent=2)
        st.download_button(
            label="📋 Download Analysis Report",
            data=report_json.encode('utf-8'),
            file_name=f'compliance_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
            mime='application/json'
        )

    with col3:
        # Agent logs export
        agent_logs = state_dict.get("agent_logs", [])
        if agent_logs:
            logs_json = pd.DataFrame(agent_logs).to_json(orient='records', indent=2)
            st.download_button(
                label="🔍 Download System Logs",
                data=logs_json.encode('utf-8'),
                file_name=f'system_logs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
                mime='application/json'
            )
        else:
            st.info("No system logs available")


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

# Initialize LangSmith
langsmith_enabled = setup_langsmith() if ENHANCED_IMPORTS else False

# Page header
st.title("🏦 Advanced Agentic AI Banking Compliance System")
st.markdown("*Enhanced Banking Compliance Analysis | CBUAE Compliant*")

# Sidebar configuration
with st.sidebar:
    st.header("🔧 System Configuration")

    # Mode selection
    st.subheader("🎯 Analysis Mode")
    analysis_mode = st.selectbox(
        "Select Analysis Mode",
        ["Enhanced (with new features)", "Legacy (backward compatible)"],
        index=0
    )
    use_enhanced = analysis_mode.startswith("Enhanced")

    # LangSmith configuration
    st.subheader("📊 LangSmith Observability")
    if langsmith_enabled:
        st.success("✅ LangSmith tracing active")
        if hasattr(config, 'langsmith_project'):
            st.info(f"Project: {config.langsmith_project}")
    else:
        st.info("ℹ️ LangSmith not configured (optional)")

    # MCP configuration
    st.subheader("🔗 MCP Integration")
    mcp_status = "✅ Enabled" if hasattr(config, 'mcp_enabled') and config.mcp_enabled else "ℹ️ Disabled"
    st.info(f"Status: {mcp_status}")

    # Analysis parameters
    st.subheader("⚙️ Analysis Parameters")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.8, 0.1)

    # System info
    st.subheader("ℹ️ System Information")
    st.info(f"Environment: {getattr(config, 'environment', 'development')}")
    st.info(f"Enhanced Features: {'Yes' if ENHANCED_IMPORTS else 'Basic'}")

# Main content
uploaded_file = st.file_uploader("Upload Banking Data (.csv)", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # Enhanced data preview with metrics
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(), use_container_width=True)

        with col2:
            st.subheader("📊 Data Summary")
            st.metric("Total Accounts", len(df))

            if 'Expected_Account_Dormant' in df.columns:
                dormant_count = len(
                    df[df['Expected_Account_Dormant'].astype(str).str.lower().isin(['yes', 'true', '1'])])
                st.metric("Pre-flagged Dormant", dormant_count)

            if 'Current_Balance' in df.columns:
                total_balance = pd.to_numeric(df['Current_Balance'], errors='coerce').sum()
                st.metric("Total Balance", f"${total_balance:,.2f}")

        # Analysis button
        if st.button("🚀 Run Advanced Compliance Analysis", type="primary"):

            # Create progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()

            with st.spinner("Initializing AI Compliance System..."):
                # Initialize orchestrator
                orchestrator = EnhancedComplianceOrchestrator(use_enhanced=use_enhanced)

                status_text.text(f"🔄 Running analysis workflow...")
                progress_bar.progress(20)

                # Run the analysis
                try:
                    final_state, agent_logs = orchestrator.run_flow(df)
                    progress_bar.progress(100)

                    st.success("✅ Compliance analysis completed!")

                    # Display results (function is now defined above)
                    display_results(final_state, agent_logs, use_enhanced, langsmith_enabled)

                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    logger.error(f"Analysis error: {e}")

                    # Show fallback options
                    st.info("💡 Try refreshing the page or checking your data format")

    except Exception as e:
        st.error(f"❌ Failed to load data: {str(e)}")
        logger.error(f"Data loading error: {e}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🏦 Advanced Agentic AI Banking Compliance System</p>
        <p>Enhanced with Modern AI Tools | CBUAE Compliant | Enterprise Ready</p>
    </div>
    """,
    unsafe_allow_html=True
)