class NotificationAgent:
    def __init__(self, memory_agent):
        self.memory_agent = memory_agent

    def run(self, state):
        try:
            notifications = []

            # Check for non-compliant data from compliance checking
            if hasattr(state, 'non_compliant_data') and not state.non_compliant_data.empty:
                notifications.append(f"Non-compliant accounts identified: {len(state.non_compliant_data)}")
                self.memory_agent.log(state.session_id, 'notification_sent', state.non_compliant_data.to_dict())

            # Check for critical compliance issues
            if hasattr(state, 'compliance_results'):
                cr = state.compliance_results

                # CB Transfer notifications
                if cr.get('transfer_candidates_cb', {}).get('count', 0) > 0:
                    notifications.append(
                        f"URGENT: {cr['transfer_candidates_cb']['count']} accounts due for CBUAE transfer")

                # Incomplete contact notifications
                if cr.get('incomplete_contact', {}).get('count', 0) > 0:
                    notifications.append(
                        f"WARNING: {cr['incomplete_contact']['count']} accounts with incomplete contact attempts")

                # Unflagged dormant notifications
                if cr.get('flag_candidates', {}).get('count', 0) > 0:
                    notifications.append(
                        f"ALERT: {cr['flag_candidates']['count']} accounts should be flagged as dormant")

            # Check for high-value dormant accounts
            if hasattr(state, 'dormant_results'):
                dr = state.dormant_results
                if dr['summary_kpis'].get('count_high_value_dormant', 0) > 0:
                    notifications.append(
                        f"HIGH VALUE ALERT: {dr['summary_kpis']['count_high_value_dormant']} high-value dormant accounts")

            if notifications:
                msg = "COMPLIANCE NOTIFICATIONS:\n" + "\n".join([f"• {n}" for n in notifications])
                self.memory_agent.log(state.session_id, 'critical_notifications', notifications)
            else:
                msg = "No critical compliance issues requiring immediate notification."
                self.memory_agent.log(state.session_id, 'notification_skipped', msg)

            state.result = msg
            state.step = 'done'

        except Exception as e:
            state.step = 'error'
            state.error = str(e)

        return state