import uuid
from agents.memory_agent import MemoryAgent
from agents.error_handler import ErrorHandlerAgent
from agents.data_processor import DataProcessor
from agents.dormancy_checker import DormancyChecker
from agents.compliance_checker import ComplianceChecker
from agents.supervisor import SupervisorAgent
from agents.notifier import NotificationAgent

class State:
    def __init__(self, session_id, data):
        self.session_id = session_id
        self.data = data
        self.step = 'data_ingest'
        self.result = None
        self.error = None

def run_flow(data):
    session_id = str(uuid.uuid4())
    memory = MemoryAgent()
    agents = {
        'data_ingest': DataProcessor(memory),
        'dormancy_check': DormancyChecker(memory),
        'compliance_check': ComplianceChecker(memory),
        'supervisor': SupervisorAgent(memory),
        'notify': NotificationAgent(memory),
        'error': ErrorHandlerAgent(memory)
    }
    state = State(session_id, data)
    while state.step != 'done':
        agent = agents[state.step]
        if state.step == 'error':
            state = agent.run(state, state.error)
            break
        else:
            state = agent.run(state)
            if state.step == 'error':
                agent = agents[state.step]
                state = agent.run(state, state.error)
                break
    return state, memory