class NotificationAgent:
    def __init__(self, memory_agent):
        self.memory_agent = memory_agent

    def run(self, state):
        try:
            msg = None
            if hasattr(state, 'non_compliant_data') and not state.non_compliant_data.empty:
                msg = "Notification sent for non-compliant accounts."
                self.memory_agent.log(state.session_id, 'notification_sent', state.non_compliant_data.to_dict())
            else:
                msg = "No non-compliant accounts to notify."
                self.memory_agent.log(state.session_id, 'notification_skipped', msg)
            state.result = msg
            state.step = 'done'
        except Exception as e:
            state.step = 'error'
            state.error = str(e)
        return state