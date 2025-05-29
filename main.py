import streamlit as st
import pandas as pd
from orchestrator import run_flow

st.title("Advanced Agentic AI Banking Compliance Demo")
st.markdown("*Comprehensive CBUAE Dormancy and Compliance Analysis*")

uploaded_file = st.file_uploader("Upload banking data (.csv)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Enhanced data preview
    st.write("### Preview of Uploaded Data")
    st.dataframe(df.head())

    # Data statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Accounts", len(df))
    with col2:
        dormant_count = len(
            df[df.get('Expected_Account_Dormant', '').astype(str).str.lower().isin(['yes', 'true', '1'])])
        st.metric("Pre-flagged Dormant", dormant_count)
    with col3:
        if 'Current_Balance' in df.columns:
            total_balance = pd.to_numeric(df['Current_Balance'], errors='coerce').sum()
            st.metric("Total Balance", f"${total_balance:,.2f}")

    if st.button("🚀 Run Advanced Compliance Flow"):
        with st.spinner("Running comprehensive dormancy and compliance analysis..."):
            state, memory = run_flow(df)

        st.success("✅ Advanced compliance analysis completed!")

        # Results summary
        st.markdown(f"**Final Result:** {state.result}")
        if state.error:
            st.error(f"❌ **Error:** {state.error}")

        # Dormant Analysis Results
        if hasattr(state, 'dormant_results'):
            st.write("## 🏦 Dormancy Analysis Results")
            dr = state.dormant_results

            # KPI metrics
            kpis = dr['summary_kpis']
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Dormant Flagged", kpis.get('total_accounts_flagged_dormant', 0))
            with col2:
                st.metric("Dormancy Rate", f"{kpis.get('percentage_dormant_of_total', 0):.1f}%")
            with col3:
                st.metric("High Value Dormant", kpis.get('count_high_value_dormant', 0))
            with col4:
                st.metric("CB Transfer Eligible", kpis.get('count_eligible_for_cb_transfer', 0))

            # Detailed breakdown
            with st.expander("📊 Detailed Dormancy Breakdown"):
                breakdown_data = {
                    'Category': [
                        'Safe Deposit Box', 'Investment Accounts', 'Fixed Deposits',
                        'Demand Deposits', 'Unclaimed Instruments'
                    ],
                    'Count': [
                        kpis.get('count_sdb_dormant', 0),
                        kpis.get('count_investment_dormant', 0),
                        kpis.get('count_fixed_deposit_dormant', 0),
                        kpis.get('count_demand_deposit_dormant', 0),
                        kpis.get('count_unclaimed_instruments', 0)
                    ]
                }
                breakdown_df = pd.DataFrame(breakdown_data)
                if breakdown_df['Count'].sum() > 0:
                    st.bar_chart(breakdown_df.set_index('Category'))
                else:
                    st.info("No dormant accounts found in detailed categories.")

        # Compliance Analysis Results
        if hasattr(state, 'compliance_results'):
            st.write("## ⚖️ Compliance Analysis Results")
            cr = state.compliance_results

            # Critical issues
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🚨 CB Transfer Due", cr.get('transfer_candidates_cb', {}).get('count', 0))
            with col2:
                st.metric("📞 Incomplete Contact", cr.get('incomplete_contact', {}).get('count', 0))
            with col3:
                st.metric("🏷️ Unflagged Dormant", cr.get('flag_candidates', {}).get('count', 0))

            # Additional compliance metrics
            with st.expander("📋 Additional Compliance Checks"):
                additional_metrics = {
                    'Internal Ledger Transfer': cr.get('ledger_candidates_internal', {}).get('count', 0),
                    'Statement Freeze Needed': cr.get('statement_freeze_needed', {}).get('count', 0)
                }

                for metric, value in additional_metrics.items():
                    if value > 0:
                        st.warning(f"**{metric}:** {value} accounts/items")
                    else:
                        st.info(f"**{metric}:** No issues found")

        # Non-compliant accounts table
        if hasattr(state, 'non_compliant_data') and not state.non_compliant_data.empty:
            st.write("### 🚨 Non-Compliant Accounts")
            st.dataframe(state.non_compliant_data)

            # Download button
            csv = state.non_compliant_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Non-Compliant Accounts",
                data=csv,
                file_name='non_compliant_accounts.csv',
                mime='text/csv'
            )
        else:
            st.info("✅ No non-compliant accounts found in this analysis.")

        # Memory log viewer
        with st.expander("🔍 View Analysis Log"):
            log_data = memory.get(state.session_id)
            for i, entry in enumerate(log_data):
                st.json({f"Step {i + 1}": entry})