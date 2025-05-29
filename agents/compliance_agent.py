from datetime import datetime, timedelta
from agents.compliance import run_all_compliance_checks


class ComplianceAgent:
    def __init__(self, memory_agent):
        self.memory_agent = memory_agent

    def run(self, state):
        try:
            # Use enhanced data from dormant identification
            data_to_check = getattr(state, 'enhanced_data', state.data)

            # Calculate threshold dates
            report_date = datetime.now()
            general_threshold_date = report_date - timedelta(days=3 * 365)  # 3 years
            freeze_threshold_date = report_date - timedelta(days=3 * 365)  # 3 years

            # Run comprehensive compliance checks
            compliance_results = run_all_compliance_checks(
                data_to_check.copy(),
                general_threshold_date=general_threshold_date,
                freeze_threshold_date=freeze_threshold_date,
                agent_name="ComplianceSystem"
            )

            # Store results in state
            state.compliance_results = compliance_results

            # Log summary
            self.memory_agent.log(
                state.session_id,
                'compliance_check',
                {
                    'total_processed': compliance_results['total_accounts_processed'],
                    'issues_found': {
                        'incomplete_contact': compliance_results['incomplete_contact']['count'],
                        'flag_candidates': compliance_results['flag_candidates']['count'],
                        'transfer_candidates': compliance_results['transfer_candidates_cb']['count']
                    }
                }
            )

            # Determine next step based on findings
            total_issues = sum([
                compliance_results['incomplete_contact']['count'],
                compliance_results['flag_candidates']['count'],
                compliance_results['transfer_candidates_cb']['count']
            ])

            if total_issues > 0:
                state.step = 'supervisor'
            else:
                state.step = 'supervisor'  # Always go to supervisor for review

        except Exception as e:
            state.step = 'error'
            state.error = str(e)

        return state