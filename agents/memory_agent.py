class MemoryAgent:
    def __init__(self):
        self.memory = {}

    def log(self, session_id, event, data):
        if session_id not in self.memory:
            self.memory[session_id] = []
        self.memory[session_id].append({'event': event, 'data': data})

    def get(self, session_id):
        return self.memory.get(session_id, [])