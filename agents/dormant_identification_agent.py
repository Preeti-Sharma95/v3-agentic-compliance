import pandas as pd
from datetime import datetime, timedelta
from agents.dormant import run_all_dormant_identification_checks


class DormantIdentificationAgent:
    def __init__(self, memory_agent):
        self.memory_agent = memory_agent

    def run(self, state):
        try:
            # Run comprehensive dormant identification
            report_date = datetime.now().strftime("%Y-%m-%d")

            dormant_results = run_all_dormant_identification_checks(
                state.data.copy(),
                report_date_str=report_date,
                dormant_flags_history_df=getattr(state, 'flags_history', None)
            )

            # Store results in state
            state.dormant_results = dormant_results

            # Log summary
            self.memory_agent.log(
                state.session_id,
                'dormant_identification',
                {
                    'total_analyzed': dormant_results['total_accounts_analyzed'],
                    'summary': dormant_results['summary_kpis']
                }
            )

            # Enhanced data with dormant flags
            state.enhanced_data = self._apply_dormant_flags(state.data, dormant_results)

            state.step = 'compliance_check'

        except Exception as e:
            state.step = 'error'
            state.error = str(e)

        return state

    def _apply_dormant_flags(self, df, dormant_results):
        """Apply dormant identification results to enhance the dataframe"""
        enhanced_df = df.copy()

        # Initialize flags if they don't exist
        required_flags = [
            'Expected_Account_Dormant',
            'Expected_Requires_Article_3_Process',
            'Expected_Transfer_to_CB_Due'
        ]

        for flag_col in required_flags:
            if flag_col not in enhanced_df.columns:
                enhanced_df[flag_col] = 'no'

        # Apply flags from dormant results
        dormant_accounts = pd.DataFrame()

        # Collect all dormant accounts
        for check_key in ['sdb_dormant', 'investment_dormant', 'fixed_deposit_dormant',
                          'demand_deposit_dormant', 'unclaimed_instruments']:
            if check_key in dormant_results and 'df' in dormant_results[check_key]:
                check_df = dormant_results[check_key]['df']
                if not check_df.empty and 'Account_ID' in check_df.columns:
                    dormant_accounts = pd.concat([dormant_accounts, check_df[['Account_ID']]])

        if not dormant_accounts.empty:
            dormant_accounts = dormant_accounts.drop_duplicates(subset=['Account_ID'])
            enhanced_df.loc[
                enhanced_df['Account_ID'].isin(dormant_accounts['Account_ID']),
                'Expected_Account_Dormant'
            ] = 'yes'

        # Apply other flags
        if 'art3_process_needed' in dormant_results and 'df' in dormant_results['art3_process_needed']:
            art3_df = dormant_results['art3_process_needed']['df']
            if not art3_df.empty and 'Account_ID' in art3_df.columns:
                enhanced_df.loc[
                    enhanced_df['Account_ID'].isin(art3_df['Account_ID']),
                    'Expected_Requires_Article_3_Process'
                ] = 'yes'

        if 'eligible_for_cb_transfer' in dormant_results and 'df' in dormant_results['eligible_for_cb_transfer']:
            cb_df = dormant_results['eligible_for_cb_transfer']['df']
            if not cb_df.empty and 'Account_ID' in cb_df.columns:
                enhanced_df.loc[
                    enhanced_df['Account_ID'].isin(cb_df['Account_ID']),
                    'Expected_Transfer_to_CB_Due'
                ] = 'yes'

        return enhanced_df
