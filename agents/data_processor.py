class DataProcessor:
    def __init__(self, memory_agent):
        self.memory_agent = memory_agent

    def run(self, state):
        data = state.data
        session_id = state.session_id
        # Check for required columns as schema validation
        if 'Account_ID' in data.columns and 'Expected_Account_Dormant' in data.columns:
            self.memory_agent.log(session_id, 'data_ingest', 'valid schema')
            state.step = 'dormancy_check'
        else:
            self.memory_agent.log(session_id, 'data_ingest', 'invalid schema')
            state.step = 'error'
            state.error = 'Invalid schema'
        return state