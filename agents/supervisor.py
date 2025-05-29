class SupervisorAgent:
    def __init__(self, memory_agent):
        self.memory_agent = memory_agent

    def run(self, state):
        try:
            # Enhanced review with dormant and compliance results
            review_summary = {"basic_review": "Supervisor reviewed the compliance and dormancy results."}

            # Add dormant identification summary
            if hasattr(state, 'dormant_results'):
                dormant_summary = state.dormant_results['summary_kpis']
                review_summary['dormant_analysis'] = {
                    'total_dormant_flagged': dormant_summary.get('total_accounts_flagged_dormant', 0),
                    'dormant_percentage': dormant_summary.get('percentage_dormant_of_total', 0),
                    'high_value_dormant': dormant_summary.get('count_high_value_dormant', 0)
                }

            # Add compliance summary
            if hasattr(state, 'compliance_results'):
                compliance_summary = {
                    'total_processed': state.compliance_results.get('total_accounts_processed', 0),
                    'critical_issues': {
                        'cb_transfer_due': state.compliance_results.get('transfer_candidates_cb', {}).get('count', 0),
                        'incomplete_contact': state.compliance_results.get('incomplete_contact', {}).get('count', 0),
                        'unflagged_dormant': state.compliance_results.get('flag_candidates', {}).get('count', 0)
                    }
                }
                review_summary['compliance_analysis'] = compliance_summary

            self.memory_agent.log(state.session_id, 'supervisor_review', review_summary)

            # Calculate confidence score based on completeness of data and analysis
            confidence_score = self._calculate_confidence_score(state)
            self.memory_agent.log(state.session_id, 'confidence_scores', {'score': confidence_score})

            state.result = f"Supervisor review completed. Confidence: {confidence_score:.2f}"
            state.step = 'notify'

        except Exception as e:
            state.step = 'error'
            state.error = str(e)

        return state

    def _calculate_confidence_score(self, state):
        """Calculate confidence score based on analysis completeness"""
        base_score = 0.7

        # Boost for dormant analysis
        if hasattr(state, 'dormant_results'):
            base_score += 0.15

        # Boost for compliance analysis
        if hasattr(state, 'compliance_results'):
            base_score += 0.15

        return min(base_score, 1.0)