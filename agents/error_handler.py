class ErrorHandlerAgent:
    def __init__(self, memory_agent):
        self.memory_agent = memory_agent

    def run(self, state, error=None):
        # Centralized error handling: log, retry, escalate (simplified)
        self.memory_agent.log(state.session_id, 'error', str(error))
        state.error = str(error)
        state.result = "Flow stopped due to error."
        state.step = 'done'
        return state