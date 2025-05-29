class DataProcessor:
    def __init__(self, memory_agent):
        self.memory_agent = memory_agent

    def run(self, state):
        data = state.data
        session_id = state.session_id

        # Enhanced schema validation
        required_cols = ['Account_ID']
        optional_cols = [
            'Expected_Account_Dormant', 'Date_Last_Cust_Initiated_Activity',
            'Account_Type', 'Current_Balance', 'Customer_ID'
        ]

        # Check required columns
        missing_required = [col for col in required_cols if col not in data.columns]
        if missing_required:
            self.memory_agent.log(session_id, 'data_ingest', f'Missing required columns: {missing_required}')
            state.step = 'error'
            state.error = f'Missing required columns: {missing_required}'
            return state

        # Log available optional columns
        available_optional = [col for col in optional_cols if col in data.columns]
        self.memory_agent.log(session_id, 'data_ingest', f'Valid schema with optional columns: {available_optional}')

        # Route to dormant identification first
        state.step = 'dormant_identification'
        return state