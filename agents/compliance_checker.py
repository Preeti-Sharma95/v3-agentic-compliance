import pandas as pd


class ComplianceChecker:
    def __init__(self, memory_agent):
        self.memory_agent = memory_agent

    def run(self, state):
        try:
            # Use enhanced compliance results if available
            if hasattr(state, 'compliance_results'):
                compliance_results = state.compliance_results

                # Extract non-compliant items
                non_compliant_items = []

                # Collect all compliance issues
                for check_name, check_result in compliance_results.items():
                    if isinstance(check_result, dict) and 'df' in check_result and 'count' in check_result:
                        if check_result['count'] > 0 and not check_result['df'].empty:
                            non_compliant_items.append(check_result['df'])

                if non_compliant_items:
                    # Combine all non-compliant data
                    state.non_compliant_data = pd.concat(non_compliant_items, ignore_index=True).drop_duplicates(
                        subset=['Account_ID'])
                    self.memory_agent.log(state.session_id, 'compliance_alert', "Non-compliant accounts flagged.")
                    state.step = 'supervisor'
                else:
                    state.step = 'supervisor'
            else:
                # Fallback to original logic
                dormant = getattr(state, 'dormant_data', None)
                if dormant is None or dormant.empty:
                    state.step = 'supervisor'
                    return state

                compliant = dormant[dormant.get('Compliant', True) == True] if 'Compliant' in dormant else dormant
                non_compliant = dormant[
                    dormant.get('Compliant', True) == False] if 'Compliant' in dormant else pd.DataFrame()

                self.memory_agent.log(state.session_id, 'compliance_result', {
                    'compliant': compliant.to_dict() if not compliant.empty else {},
                    'non_compliant': non_compliant.to_dict() if not non_compliant.empty else {}
                })

                if not non_compliant.empty:
                    state.non_compliant_data = non_compliant
                    self.memory_agent.log(state.session_id, 'compliance_alert', "Non-compliant accounts flagged.")
                    state.step = 'supervisor'
                else:
                    state.compliant_data = compliant
                    state.step = 'supervisor'

        except Exception as e:
            state.step = 'error'
            state.error = str(e)

        return state