class ComplianceChecker:
    def __init__(self, memory_agent):
        self.memory_agent = memory_agent

    def run(self, state):
        try:
            dormant = getattr(state, 'dormant_data', None)
            # If no dormant data, skip
            if dormant is None or dormant.empty:
                state.step = 'supervisor'
                return state

            compliant = dormant[dormant['Compliant'] == True] if 'Compliant' in dormant else dormant
            non_compliant = dormant[dormant['Compliant'] == False] if 'Compliant' in dormant else dormant.iloc[0:0]

            self.memory_agent.log(state.session_id, 'compliance_result', {
                'compliant': compliant.to_dict(),
                'non_compliant': non_compliant.to_dict()
            })

            if not non_compliant.empty:
                state.non_compliant_data = non_compliant
                self.memory_agent.log(state.session_id, 'compliance_alert', "Non-compliant accounts flagged.")
                state.step = 'notify'
            else:
                state.compliant_data = compliant
                state.step = 'supervisor'
        except Exception as e:
            state.step = 'error'
            state.error = str(e)
        return state