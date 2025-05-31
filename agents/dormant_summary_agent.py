# agents/dormant_summary_agent.py
"""
Comprehensive Dormant Account Summary Agent
Orchestrates all dormancy analyzers and provides consolidated reporting
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging
from dataclasses import dataclass, asdict

from agents.dormant_analyzers import (
    SafeDepositDormancyAnalyzer,
    InvestmentInactivityAnalyzer,
    FixedDepositInactivityAnalyzer,
    DemandDepositInactivityAnalyzer,
    UnclaimedPaymentInstrumentsAnalyzer,
    CBUAETransferEligibilityAnalyzer,
    Article3ProcessAnalyzer,
    ProactiveContactAnalyzer,
    HighValueDormantAnalyzer,
    DormantToActiveTransitionAnalyzer,
    DormancyConfig
)

logger = logging.getLogger(__name__)


@dataclass
class DormancyAnalysisResult:
    """Structured result for individual dormancy analysis"""
    analyzer_name: str
    regulation_article: str
    flagged_accounts: pd.DataFrame
    count: int
    description: str
    details: Dict[str, Any]
    priority_level: str
    risk_score: float


@dataclass
class DormancySummaryReport:
    """Comprehensive dormancy summary report"""
    report_date: str
    total_accounts_analyzed: int
    summary_statistics: Dict[str, Any]
    analysis_results: List[DormancyAnalysisResult]
    consolidated_findings: Dict[str, Any]
    action_recommendations: List[Dict[str, Any]]
    regulatory_compliance_status: Dict[str, Any]
    risk_assessment: Dict[str, Any]


class DormantSummaryAgent:
    """
    Master agent that orchestrates all dormancy analyzers and provides
    comprehensive summary reporting with actionable insights
    """

    def __init__(self, memory_agent=None):
        self.memory_agent = memory_agent
        self.config = DormancyConfig()

        # Initialize all analyzers
        self.analyzers = {
            'safe_deposit_dormancy': SafeDepositDormancyAnalyzer(memory_agent),
            'investment_inactivity': InvestmentInactivityAnalyzer(memory_agent),
            'fixed_deposit_inactivity': FixedDepositInactivityAnalyzer(memory_agent),
            'demand_deposit_inactivity': DemandDepositInactivityAnalyzer(memory_agent),
            'unclaimed_payment_instruments': UnclaimedPaymentInstrumentsAnalyzer(memory_agent),
            'cbuae_transfer_eligibility': CBUAETransferEligibilityAnalyzer(memory_agent),
            'article3_process': Article3ProcessAnalyzer(memory_agent),
            'proactive_contact': ProactiveContactAnalyzer(memory_agent),
            'high_value_dormant': HighValueDormantAnalyzer(memory_agent),
            'dormant_to_active_transitions': DormantToActiveTransitionAnalyzer(memory_agent)
        }

        # Priority mapping for different types of findings
        self.priority_mapping = {
            'cbuae_transfer_eligibility': 'CRITICAL',
            'high_value_dormant': 'HIGH',
            'safe_deposit_dormancy': 'HIGH',
            'article3_process': 'MEDIUM',
            'investment_inactivity': 'MEDIUM',
            'fixed_deposit_inactivity': 'MEDIUM',
            'demand_deposit_inactivity': 'MEDIUM',
            'unclaimed_payment_instruments': 'MEDIUM',
            'proactive_contact': 'LOW',
            'dormant_to_active_transitions': 'LOW'
        }

    def run(self, state):
        """Main execution method for the agent workflow"""
        try:
            # Generate comprehensive dormancy analysis
            report_date = datetime.now()
            summary_report = self.generate_comprehensive_analysis(
                state.data.copy(),
                report_date,
                getattr(state, 'flags_history', None),
                state.session_id
            )

            # Store results in state
            state.dormant_summary_report = summary_report
            state.enhanced_data = self._apply_consolidated_flags(state.data, summary_report)

            # Log comprehensive summary
            self.memory_agent.log(
                state.session_id,
                'dormant_summary_analysis',
                {
                    'total_analyzed': summary_report.total_accounts_analyzed,
                    'critical_findings': len(
                        [r for r in summary_report.analysis_results if r.priority_level == 'CRITICAL']),
                    'high_priority_findings': len(
                        [r for r in summary_report.analysis_results if r.priority_level == 'HIGH']),
                    'overall_risk_score': summary_report.risk_assessment.get('overall_risk_score', 0)
                }
            )

            state.step = 'compliance_check'

        except Exception as e:
            logger.error(f"Dormant Summary Agent Error: {e}")
            state.step = 'error'
            state.error = str(e)

        return state

    def generate_comprehensive_analysis(self,
                                        df: pd.DataFrame,
                                        report_date: datetime,
                                        dormant_history_df: Optional[pd.DataFrame] = None,
                                        session_id: str = None) -> DormancySummaryReport:
        """
        Generate comprehensive dormancy analysis using all analyzers
        """
        logger.info(f"Starting comprehensive dormancy analysis for {len(df)} accounts")

        analysis_results = []
        all_flagged_accounts = pd.DataFrame()

        # Run all analyzers
        for analyzer_name, analyzer in self.analyzers.items():
            try:
                logger.debug(f"Running {analyzer_name} analysis")

                if analyzer_name == 'dormant_to_active_transitions':
                    # Special handling for transition analyzer
                    flagged_df, count, description, details = analyzer.analyze(
                        df, report_date, dormant_history_df
                    )
                else:
                    flagged_df, count, description, details = analyzer.analyze(df, report_date)

                # Calculate risk score based on count and priority
                risk_score = self._calculate_risk_score(count, analyzer_name, details)
                priority_level = self.priority_mapping.get(analyzer_name, 'MEDIUM')

                # Create structured result
                result = DormancyAnalysisResult(
                    analyzer_name=analyzer_name,
                    regulation_article=details.get('regulation', 'N/A'),
                    flagged_accounts=flagged_df,
                    count=count,
                    description=description,
                    details=details,
                    priority_level=priority_level,
                    risk_score=risk_score
                )

                analysis_results.append(result)

                # Collect all flagged accounts for consolidation
                if not flagged_df.empty and 'Account_ID' in flagged_df.columns:
                    flagged_df_with_source = flagged_df.copy()
                    flagged_df_with_source['Flagged_By'] = analyzer_name
                    all_flagged_accounts = pd.concat([all_flagged_accounts, flagged_df_with_source], ignore_index=True)

                # Log individual analysis
                if self.memory_agent and session_id:
                    analyzer.log_analysis(session_id, analyzer_name, {
                        'count': count,
                        'description': description,
                        'priority': priority_level,
                        'risk_score': risk_score
                    })

            except Exception as e:
                logger.error(f"Error in {analyzer_name} analysis: {e}")
                # Continue with other analyzers
                continue

        # Generate consolidated findings
        consolidated_findings = self._generate_consolidated_findings(analysis_results, all_flagged_accounts)

        # Generate summary statistics
        summary_stats = self._generate_summary_statistics(df, analysis_results, all_flagged_accounts)

        # Generate action recommendations
        action_recommendations = self._generate_action_recommendations(analysis_results)

        # Assess regulatory compliance
        compliance_status = self._assess_regulatory_compliance(analysis_results)

        # Calculate overall risk assessment
        risk_assessment = self._calculate_overall_risk_assessment(analysis_results, summary_stats)

        # Create comprehensive report
        summary_report = DormancySummaryReport(
            report_date=report_date.strftime("%Y-%m-%d %H:%M:%S"),
            total_accounts_analyzed=len(df),
            summary_statistics=summary_stats,
            analysis_results=analysis_results,
            consolidated_findings=consolidated_findings,
            action_recommendations=action_recommendations,
            regulatory_compliance_status=compliance_status,
            risk_assessment=risk_assessment
        )

        logger.info(
            f"Comprehensive dormancy analysis completed. Found {len(all_flagged_accounts)} flagged accounts across {len(analysis_results)} checks")

        return summary_report

    def _calculate_risk_score(self, count: int, analyzer_name: str, details: Dict[str, Any]) -> float:
        """Calculate risk score for an analysis result"""
        base_score = min(count / 100, 1.0)  # Normalize by 100 accounts

        # Apply multipliers based on analyzer type
        multipliers = {
            'cbuae_transfer_eligibility': 2.0,
            'high_value_dormant': 1.8,
            'safe_deposit_dormancy': 1.5,
            'article3_process': 1.2,
            'proactive_contact': 0.5
        }

        multiplier = multipliers.get(analyzer_name, 1.0)

        # Consider balance for high-value accounts
        if 'total_balance' in details and details['total_balance'] > 1000000:
            multiplier *= 1.5

        return min(base_score * multiplier, 1.0)

    def _generate_consolidated_findings(self, analysis_results: List[DormancyAnalysisResult],
                                        all_flagged_accounts: pd.DataFrame) -> Dict[str, Any]:
        """Generate consolidated findings across all analyses"""

        # Remove duplicates based on Account_ID
        unique_flagged_accounts = all_flagged_accounts.drop_duplicates(
            subset=['Account_ID']) if not all_flagged_accounts.empty else pd.DataFrame()

        # Account overlap analysis
        overlap_analysis = {}
        if not all_flagged_accounts.empty:
            account_flag_counts = all_flagged_accounts['Account_ID'].value_counts()
            overlap_analysis = {
                'accounts_flagged_multiple_times': len(account_flag_counts[account_flag_counts > 1]),
                'max_flags_per_account': account_flag_counts.max() if len(account_flag_counts) > 0 else 0,
                'most_flagged_accounts': account_flag_counts.head(5).to_dict()
            }

        # Priority distribution
        priority_counts = {}
        for result in analysis_results:
            if result.count > 0:
                priority_counts[result.priority_level] = priority_counts.get(result.priority_level, 0) + result.count

        # Account type distribution
        account_type_analysis = {}
        if not unique_flagged_accounts.empty and 'Account_Type' in unique_flagged_accounts.columns:
            account_type_analysis = unique_flagged_accounts['Account_Type'].value_counts().head(10).to_dict()

        return {
            'total_unique_flagged_accounts': len(unique_flagged_accounts),
            'total_flags_across_all_checks': len(all_flagged_accounts),
            'account_overlap_analysis': overlap_analysis,
            'priority_distribution': priority_counts,
            'account_type_distribution': account_type_analysis,
            'critical_issues_count': len(
                [r for r in analysis_results if r.priority_level == 'CRITICAL' and r.count > 0]),
            'high_priority_issues_count': len(
                [r for r in analysis_results if r.priority_level == 'HIGH' and r.count > 0])
        }

    def _generate_summary_statistics(self, df: pd.DataFrame,
                                     analysis_results: List[DormancyAnalysisResult],
                                     all_flagged_accounts: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for the analysis"""

        total_accounts = len(df)
        unique_flagged = len(
            all_flagged_accounts.drop_duplicates(subset=['Account_ID'])) if not all_flagged_accounts.empty else 0

        # Calculate total balance at risk
        total_balance_at_risk = 0
        if not all_flagged_accounts.empty and 'Current_Balance' in all_flagged_accounts.columns:
            unique_accounts = all_flagged_accounts.drop_duplicates(subset=['Account_ID'])
            total_balance_at_risk = pd.to_numeric(unique_accounts['Current_Balance'], errors='coerce').sum()

        # Dormancy rate by regulation
        regulation_breakdown = {}
        for result in analysis_results:
            if result.count > 0:
                regulation_breakdown[result.regulation_article] = {
                    'count': result.count,
                    'percentage': round((result.count / total_accounts) * 100, 2) if total_accounts > 0 else 0
                }

        return {
            'total_accounts_processed': total_accounts,
            'total_flagged_accounts': unique_flagged,
            'overall_dormancy_rate': round((unique_flagged / total_accounts) * 100, 2) if total_accounts > 0 else 0,
            'total_balance_at_risk': total_balance_at_risk,
            'regulation_breakdown': regulation_breakdown,
            'analysis_coverage': len([r for r in analysis_results if r.count >= 0]),  # Successful analyses
            'failed_analyses': len([r for r in analysis_results if 'Error' in r.description])
        }

    def _generate_action_recommendations(self, analysis_results: List[DormancyAnalysisResult]) -> List[Dict[str, Any]]:
        """Generate prioritized action recommendations"""

        recommendations = []

        # Critical actions first
        critical_results = [r for r in analysis_results if r.priority_level == 'CRITICAL' and r.count > 0]
        for result in critical_results:
            if 'cbuae_transfer' in result.analyzer_name:
                recommendations.append({
                    'priority': 'CRITICAL',
                    'action': 'Immediate CBUAE Transfer Processing',
                    'description': f'Process {result.count} accounts for immediate transfer to Central Bank',
                    'deadline': 'Within 30 days',
                    'regulation': result.regulation_article,
                    'affected_accounts': result.count
                })

        # High priority actions
        high_results = [r for r in analysis_results if r.priority_level == 'HIGH' and r.count > 0]
        for result in high_results:
            if 'high_value' in result.analyzer_name:
                recommendations.append({
                    'priority': 'HIGH',
                    'action': 'Enhanced Monitoring Setup',
                    'description': f'Implement enhanced monitoring for {result.count} high-value dormant accounts',
                    'deadline': 'Within 60 days',
                    'regulation': 'Internal Risk Management',
                    'affected_accounts': result.count
                })
            elif 'safe_deposit' in result.analyzer_name:
                recommendations.append({
                    'priority': 'HIGH',
                    'action': 'Safe Deposit Box Review',
                    'description': f'Review and process {result.count} Safe Deposit Boxes with unpaid charges',
                    'deadline': 'Within 90 days',
                    'regulation': result.regulation_article,
                    'affected_accounts': result.count
                })

        # Medium priority actions
        medium_results = [r for r in analysis_results if r.priority_level == 'MEDIUM' and r.count > 0]
        article3_count = sum(r.count for r in medium_results if 'article3' in r.analyzer_name)
        if article3_count > 0:
            recommendations.append({
                'priority': 'MEDIUM',
                'action': 'Article 3 Process Implementation',
                'description': f'Complete Article 3 compliance process for {article3_count} dormant accounts',
                'deadline': 'Within 120 days',
                'regulation': 'Article 3 CBUAE',
                'affected_accounts': article3_count
            })

        # Preventive actions
        proactive_results = [r for r in analysis_results if 'proactive' in r.analyzer_name and r.count > 0]
        for result in proactive_results:
            recommendations.append({
                'priority': 'LOW',
                'action': 'Proactive Customer Outreach',
                'description': f'Contact {result.count} customers to prevent dormancy',
                'deadline': 'Within 180 days',
                'regulation': 'Preventive Compliance',
                'affected_accounts': result.count
            })

        return recommendations

    def _assess_regulatory_compliance(self, analysis_results: List[DormancyAnalysisResult]) -> Dict[str, Any]:
        """Assess overall regulatory compliance status"""

        compliance_status = {
            'overall_status': 'COMPLIANT',
            'critical_violations': 0,
            'regulatory_gaps': [],
            'compliance_score': 100.0
        }

        # Check for critical violations
        for result in analysis_results:
            if result.priority_level == 'CRITICAL' and result.count > 0:
                compliance_status['critical_violations'] += result.count
                compliance_status['overall_status'] = 'NON_COMPLIANT'
                compliance_status['regulatory_gaps'].append({
                    'regulation': result.regulation_article,
                    'issue': result.description,
                    'severity': 'CRITICAL'
                })

        # Calculate compliance score
        total_issues = sum(r.count for r in analysis_results if r.count > 0)
        critical_weight = 20
        high_weight = 10
        medium_weight = 5

        penalty = 0
        for result in analysis_results:
            if result.count > 0:
                if result.priority_level == 'CRITICAL':
                    penalty += result.count * critical_weight
                elif result.priority_level == 'HIGH':
                    penalty += result.count * high_weight
                elif result.priority_level == 'MEDIUM':
                    penalty += result.count * medium_weight

        compliance_status['compliance_score'] = max(0, 100 - penalty)

        if compliance_status['compliance_score'] < 70:
            compliance_status['overall_status'] = 'NON_COMPLIANT'
        elif compliance_status['compliance_score'] < 85:
            compliance_status['overall_status'] = 'NEEDS_ATTENTION'

        return compliance_status

    def _calculate_overall_risk_assessment(self, analysis_results: List[DormancyAnalysisResult],
                                           summary_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall risk assessment"""

        # Aggregate risk scores
        total_risk_score = sum(r.risk_score for r in analysis_results)
        avg_risk_score = total_risk_score / len(analysis_results) if analysis_results else 0

        # Risk level determination
        if avg_risk_score >= 0.8:
            risk_level = 'CRITICAL'
        elif avg_risk_score >= 0.6:
            risk_level = 'HIGH'
        elif avg_risk_score >= 0.4:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'

        # Financial risk assessment
        financial_risk = 'LOW'
        balance_at_risk = summary_stats.get('total_balance_at_risk', 0)
        if balance_at_risk > 10000000:  # 10M
            financial_risk = 'CRITICAL'
        elif balance_at_risk > 5000000:  # 5M
            financial_risk = 'HIGH'
        elif balance_at_risk > 1000000:  # 1M
            financial_risk = 'MEDIUM'

        return {
            'overall_risk_score': round(avg_risk_score, 3),
            'risk_level': risk_level,
            'financial_risk': financial_risk,
            'balance_at_risk': balance_at_risk,
            'regulatory_risk': 'HIGH' if any(r.priority_level == 'CRITICAL' for r in analysis_results) else 'MEDIUM',
            'trend_analysis': 'STABLE',  # Would require historical data
            'risk_factors': [
                f"{r.analyzer_name}: {r.risk_score:.2f}" for r in analysis_results
                if r.risk_score > 0.5
            ]
        }

    def _apply_consolidated_flags(self, df: pd.DataFrame,
                                  summary_report: DormancySummaryReport) -> pd.DataFrame:
        """Apply consolidated dormancy flags to the dataframe"""

        enhanced_df = df.copy()

        # Initialize flags if they don't exist
        flag_columns = [
            'Expected_Account_Dormant',
            'Expected_Requires_Article_3_Process',
            'Expected_Transfer_to_CB_Due',
            'High_Value_Dormant',
            'Proactive_Contact_Required'
        ]

        for col in flag_columns:
            if col not in enhanced_df.columns:
                enhanced_df[col] = 'no'

        # Apply flags based on analysis results
        for result in summary_report.analysis_results:
            if result.count > 0 and not result.flagged_accounts.empty:
                account_ids = result.flagged_accounts['Account_ID'].tolist()

                # Set appropriate flags based on analyzer type
                if 'dormancy' in result.analyzer_name or 'inactivity' in result.analyzer_name or 'unclaimed' in result.analyzer_name:
                    enhanced_df.loc[enhanced_df['Account_ID'].isin(account_ids), 'Expected_Account_Dormant'] = 'yes'

                if 'article3' in result.analyzer_name:
                    enhanced_df.loc[
                        enhanced_df['Account_ID'].isin(account_ids), 'Expected_Requires_Article_3_Process'] = 'yes'

                if 'cbuae_transfer' in result.analyzer_name:
                    enhanced_df.loc[enhanced_df['Account_ID'].isin(account_ids), 'Expected_Transfer_to_CB_Due'] = 'yes'

                if 'high_value' in result.analyzer_name:
                    enhanced_df.loc[enhanced_df['Account_ID'].isin(account_ids), 'High_Value_Dormant'] = 'yes'

                if 'proactive_contact' in result.analyzer_name:
                    enhanced_df.loc[enhanced_df['Account_ID'].isin(account_ids), 'Proactive_Contact_Required'] = 'yes'

        return enhanced_df

    def generate_executive_summary(self, summary_report: DormancySummaryReport) -> str:
        """Generate executive summary text"""

        critical_count = len(
            [r for r in summary_report.analysis_results if r.priority_level == 'CRITICAL' and r.count > 0])
        high_count = len([r for r in summary_report.analysis_results if r.priority_level == 'HIGH' and r.count > 0])

        summary = f"""
DORMANT ACCOUNT ANALYSIS - EXECUTIVE SUMMARY
Report Date: {summary_report.report_date}

OVERVIEW:
• Total Accounts Analyzed: {summary_report.total_accounts_analyzed:,}
• Flagged Accounts: {summary_report.consolidated_findings['total_unique_flagged_accounts']:,}
• Overall Dormancy Rate: {summary_report.summary_statistics['overall_dormancy_rate']:.2f}%
• Total Balance at Risk: AED {summary_report.summary_statistics['total_balance_at_risk']:,.2f}

RISK ASSESSMENT:
• Overall Risk Level: {summary_report.risk_assessment['risk_level']}
• Financial Risk: {summary_report.risk_assessment['financial_risk']}
• Compliance Status: {summary_report.regulatory_compliance_status['overall_status']}
• Compliance Score: {summary_report.regulatory_compliance_status['compliance_score']:.1f}/100

CRITICAL FINDINGS:
• Critical Issues: {critical_count}
• High Priority Issues: {high_count}
• Immediate Actions Required: {len([r for r in summary_report.action_recommendations if r['priority'] == 'CRITICAL'])}

REGULATORY COMPLIANCE:
• CBUAE Transfer Due: {sum(r.count for r in summary_report.analysis_results if 'cbuae_transfer' in r.analyzer_name)}
• Article 3 Process Required: {sum(r.count for r in summary_report.analysis_results if 'article3' in r.analyzer_name)}
• High-Value Accounts: {sum(r.count for r in summary_report.analysis_results if 'high_value' in r.analyzer_name)}
        """

        return summary.strip()