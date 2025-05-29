class DormancyChecker:
    def __init__(self, memory_agent):
        self.memory_agent = memory_agent

    def run(self, state):
        try:
            dormant = state.data[state.data['Expected_Account_Dormant'] == 'Yes']
            self.memory_agent.log(state.session_id, 'dormancy_flag', dormant.to_dict())
            if not dormant.empty:
                state.dormant_data = dormant
                self.memory_agent.log(state.session_id, 'dormancy_insights', "Dormant accounts found and stored.")
                state.step = 'compliance_check'
            else:
                state.step = 'supervisor'
        except Exception as e:
            state.step = 'error'
            state.error = str(e)
        return state