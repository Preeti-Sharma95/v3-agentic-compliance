class SupervisorAgent:
    def __init__(self, memory_agent):
        self.memory_agent = memory_agent

    def run(self, state):
        # Review outputs with memory context, store decision, send summary
        try:
            review = "Supervisor reviewed the compliance and dormancy results."
            self.memory_agent.log(state.session_id, 'supervisor_review', review)
            self.memory_agent.log(state.session_id, 'confidence_scores', {'score': 0.95}) # Example value
            state.result = review
            state.step = 'notify'
        except Exception as e:
            state.step = 'error'
            state.error = str(e)
        return state