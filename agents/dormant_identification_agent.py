# agents/dormant_identification_agent.py
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import logging

# Import the new modular system
try:
    from agents.dormant_summary_agent import DormantSummaryAgent

    ENHANCED_DORMANT_AVAILABLE = True
except ImportError:
    ENHANCED_DORMANT_AVAILABLE = False

# Import legacy system for fallback
from agents.dormant import run_all_dormant_identification_checks

logger = logging.getLogger(__name__)


class DormantIdentificationAgent:
    """
    Enhanced Dormant Identification Agent that can use either the new modular
    analyzer system or fall back to the legacy system
    """

    def __init__(self, memory_agent):
        self.memory_agent = memory_agent

        # Initialize enhanced system if available
        if ENHANCED_DORMANT_AVAILABLE:
            try:
                self.dormant_summary_agent = DormantSummaryAgent(memory_agent)
                self.enhanced_mode = True
                logger.info("Dormant Identification Agent initialized with enhanced modular analyzers")
            except Exception as e:
                logger.warning(f"Failed to initialize enhanced mode: {e}")
                self.enhanced_mode = False
                self.dormant_summary_agent = None
        else:
            logger.info("Enhanced dormant analyzers not available, using legacy mode")
            self.enhanced_mode = False
            self.dormant_summary_agent = None

    def run(self, state):
        try:
            logger.info(f"Starting dormant identification for session {state.session_id}")

            if self.enhanced_mode and self.dormant_summary_agent:
                return self._run_enhanced_analysis(state)
            else:
                return self._run_legacy_analysis(state)

        except Exception as e:
            logger.error(f"Dormant Identification Agent Error: {e}")
            state.step = 'error'
            state.error = str(e)
            return state

    def _run_enhanced_analysis(self, state):
        """Run enhanced analysis using the new modular system"""
        try:
            # Generate comprehensive dormancy analysis using the summary agent
            report_date = datetime.now()

            # Run the comprehensive analysis
            summary_report = self.dormant_summary_agent.generate_comprehensive_analysis(
                df=state.data.copy(),
                report_date=report_date,
                dormant_history_df=getattr(state, 'flags_history', None),
                session_id=state.session_id
            )

            # Store comprehensive results in state
            state.dormant_summary_report = summary_report
            state.dormant_results = self._convert_to_legacy_format(summary_report)

            # Apply enhanced flags to the data
            state.enhanced_data = self.dormant_summary_agent._apply_consolidated_flags(
                state.data, summary_report
            )

            # Generate executive summary
            executive_summary = self.dormant_summary_agent.generate_executive_summary(summary_report)
            state.executive_summary = executive_summary

            # Log comprehensive summary to memory
            self._log_enhanced_results(state.session_id, summary_report)

            # Determine next step based on findings
            if self._has_critical_findings(summary_report):
                logger.warning(f"Critical dormancy findings detected for session {state.session_id}")
                state.step = 'compliance_check'
            elif self._has_any_findings(summary_report):
                logger.info(f"Dormancy findings detected for session {state.session_id}")
                state.step = 'compliance_check'
            else:
                logger.info(f"No significant dormancy findings for session {state.session_id}")
                state.step = 'supervisor'

            return state

        except Exception as e:
            logger.error(f"Enhanced analysis failed, falling back to legacy: {e}")
            return self._run_legacy_analysis(state)

    def _run_legacy_analysis(self, state):
        """Run legacy analysis using the original dormant.py system"""
        try:
            # Run comprehensive dormant identification using legacy system
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
                    'summary': dormant_results['summary_kpis'],
                    'analysis_mode': 'legacy'
                }
            )

            # Enhanced data with dormant flags
            state.enhanced_data = self._apply_dormant_flags(state.data, dormant_results)

            state.step = 'compliance_check'
            return state

        except Exception as e:
            logger.error(f"Legacy analysis also failed: {e}")
            state.step = 'error'
            state.error = str(e)
            return state

    def _convert_to_legacy_format(self, summary_report) -> Dict[str, Any]:
        """Convert new summary report format to legacy format for backward compatibility"""
        legacy_results = {
            "report_date_used": summary_report.report_date.split()[0],  # Just the date part
            "total_accounts_analyzed": summary_report.total_accounts_analyzed,
            "summary_kpis": {}
        }

        # Convert each analysis result to legacy format
        for result in summary_report.analysis_results:
            analyzer_key = result.analyzer_name

            legacy_results[analyzer_key] = {
                "df": result.flagged_accounts,
                "count": result.count,
                "desc": result.description,
                "details": result.details
            }

        # Generate summary KPIs in legacy format
        summary_kpis = {
            "total_accounts_flagged_dormant": summary_report.consolidated_findings.get('total_unique_flagged_accounts',
                                                                                       0),
            "percentage_dormant_of_total": summary_report.summary_statistics.get('overall_dormancy_rate', 0),
            "total_dormant_balance_aed": summary_report.summary_statistics.get('total_balance_at_risk', 0)
        }

        # Add individual counts for backward compatibility
        for result in summary_report.analysis_results:
            if 'safe_deposit' in result.analyzer_name:
                summary_kpis["count_sdb_dormant"] = result.count
            elif 'investment' in result.analyzer_name:
                summary_kpis["count_investment_dormant"] = result.count
            elif 'fixed_deposit' in result.analyzer_name:
                summary_kpis["count_fixed_deposit_dormant"] = result.count
            elif 'demand_deposit' in result.analyzer_name:
                summary_kpis["count_demand_deposit_dormant"] = result.count
            elif 'unclaimed' in result.analyzer_name:
                summary_kpis["count_unclaimed_instruments"] = result.count
            elif 'cbuae_transfer' in result.analyzer_name:
                summary_kpis["count_eligible_for_cb_transfer"] = result.count
            elif 'high_value' in result.analyzer_name:
                summary_kpis["count_high_value_dormant"] = result.count
            elif 'transitions' in result.analyzer_name:
                summary_kpis["count_dormant_to_active_transitions"] = result.count
            elif 'article3' in result.analyzer_name:
                summary_kpis["count_needing_art3_process"] = result.count
            elif 'proactive_contact' in result.analyzer_name:
                summary_kpis["count_needing_proactive_contact"] = result.count

        legacy_results["summary_kpis"] = summary_kpis

        return legacy_results

    def _log_enhanced_results(self, session_id: str, summary_report):
        """Log enhanced results to memory with detailed insights"""

        # Log overall summary
        self.memory_agent.log(
            session_id,
            'dormant_identification_summary',
            {
                'total_analyzed': summary_report.total_accounts_analyzed,
                'total_flagged': summary_report.consolidated_findings['total_unique_flagged_accounts'],
                'dormancy_rate': summary_report.summary_statistics['overall_dormancy_rate'],
                'risk_level': summary_report.risk_assessment['risk_level'],
                'compliance_status': summary_report.regulatory_compliance_status['overall_status'],
                'analysis_mode': 'enhanced'
            },
            importance=0.9
        )

        # Log critical findings
        critical_results = [r for r in summary_report.analysis_results if
                            r.priority_level == 'CRITICAL' and r.count > 0]
        if critical_results:
            self.memory_agent.log(
                session_id,
                'critical_dormancy_findings',
                {
                    'critical_count': len(critical_results),
                    'findings': [
                        {
                            'analyzer': r.analyzer_name,
                            'regulation': r.regulation_article,
                            'count': r.count,
                            'description': r.description
                        } for r in critical_results
                    ]
                },
                importance=1.0
            )

    def _has_critical_findings(self, summary_report) -> bool:
        """Check if there are any critical findings requiring immediate attention"""
        return any(
            result.priority_level == 'CRITICAL' and result.count > 0
            for result in summary_report.analysis_results
        )

    def _has_any_findings(self, summary_report) -> bool:
        """Check if there are any findings at all"""
        return summary_report.consolidated_findings['total_unique_flagged_accounts'] > 0

    def _apply_dormant_flags(self, df, dormant_results):
        """Apply dormant identification results to enhance the dataframe (legacy method)"""
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

    def get_analysis_mode(self) -> str:
        """Get current analysis mode"""
        return "enhanced" if self.enhanced_mode else "legacy"

    def get_available_analyzers(self) -> list:
        """Get list of available analyzers"""
        if self.enhanced_mode and self.dormant_summary_agent:
            return list(self.dormant_summary_agent.analyzers.keys())
        else:
            return [
                'sdb_dormant', 'investment_dormant', 'fixed_deposit_dormant',
                'demand_deposit_dormant', 'unclaimed_instruments', 'eligible_for_cb_transfer',
                'art3_process_needed', 'proactive_contact_needed', 'high_value_dormant',
                'dormant_to_active'
            ]