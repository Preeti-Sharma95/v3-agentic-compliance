# orchestrator.py
import uuid
import logging
from agents.memory_agent import MemoryAgent
from agents.error_handler import ErrorHandlerAgent
from agents.data_processor import DataProcessor
from agents.dormant_identification_agent import DormantIdentificationAgent
from agents.compliance_agent import ComplianceAgent
from agents.dormancy_checker import DormancyChecker
from agents.compliance_checker import ComplianceChecker
from agents.supervisor import SupervisorAgent
from agents.notifier import NotificationAgent

logger = logging.getLogger(__name__)


class State:
    def __init__(self, session_id, data):
        self.session_id = session_id
        self.data = data
        self.step = 'data_ingest'
        self.result = None
        self.error = None

        # Enhanced attributes for new system
        self.enhanced_data = None
        self.dormant_results = None
        self.dormant_summary_report = None
        self.executive_summary = None
        self.compliance_results = None
        self.non_compliant_data = None


def run_flow(data):
    """
    Run the banking compliance workflow with enhanced dormancy analysis
    """
    session_id = str(uuid.uuid4())
    memory = MemoryAgent()

    logger.info(f"Starting compliance workflow for session: {session_id}")

    # Enhanced agent registry with new dormant and compliance agents
    agents = {
        'data_ingest': DataProcessor(memory),
        'dormant_identification': DormantIdentificationAgent(memory),
        'compliance_check': ComplianceAgent(memory),
        'dormancy_check': DormancyChecker(memory),  # For backward compatibility
        'compliance_checker': ComplianceChecker(memory),  # Additional compliance checking
        'supervisor': SupervisorAgent(memory),
        'notify': NotificationAgent(memory),
        'error': ErrorHandlerAgent(memory)
    }

    state = State(session_id, data)

    # Log initial state
    memory.log(session_id, 'workflow_start', {
        'data_shape': data.shape if hasattr(data, 'shape') else len(data),
        'initial_step': state.step
    })

    step_count = 0
    max_steps = 20  # Prevent infinite loops

    while state.step != 'done' and step_count < max_steps:
        step_count += 1

        logger.info(f"Processing step: {state.step} (step {step_count})")

        agent = agents.get(state.step)
        if not agent:
            logger.error(f"Unknown step: {state.step}")
            state.step = 'error'
            state.error = f"Unknown step: {state.step}"
            agent = agents['error']

        try:
            if state.step == 'error':
                state = agent.run(state, state.error)
                break
            else:
                state = agent.run(state)

                # Log step completion
                memory.log(session_id, f'step_completed_{state.step}', {
                    'step_number': step_count,
                    'next_step': state.step if hasattr(state, 'step') else 'unknown'
                })

                if state.step == 'error':
                    logger.error(f"Error in step {step_count}: {state.error}")
                    agent = agents[state.step]
                    state = agent.run(state, state.error)
                    break

        except Exception as e:
            logger.error(f"Exception in step {state.step}: {e}")
            state.step = 'error'
            state.error = str(e)
            agent = agents['error']
            state = agent.run(state, state.error)
            break

    if step_count >= max_steps:
        logger.error(f"Workflow exceeded maximum steps ({max_steps})")
        state.step = 'error'
        state.error = "Workflow exceeded maximum steps"

    # Log workflow completion
    memory.log(session_id, 'workflow_complete', {
        'final_step': state.step,
        'total_steps': step_count,
        'has_error': bool(state.error),
        'has_result': bool(state.result)
    })

    logger.info(f"Workflow completed for session: {session_id} in {step_count} steps")

    return state, memory


def run_enhanced_flow(data, config=None):
    """
    Run enhanced workflow with additional configuration options
    """
    session_id = str(uuid.uuid4())
    memory = MemoryAgent()

    logger.info(f"Starting enhanced compliance workflow for session: {session_id}")

    # Enhanced agent initialization with configuration
    try:
        dormant_agent = DormantIdentificationAgent(memory)
        analysis_mode = dormant_agent.get_analysis_mode()

        logger.info(f"Dormant analysis mode: {analysis_mode}")

        memory.log(session_id, 'enhanced_workflow_start', {
            'analysis_mode': analysis_mode,
            'available_analyzers': dormant_agent.get_available_analyzers(),
            'data_shape': data.shape if hasattr(data, 'shape') else len(data)
        })

    except Exception as e:
        logger.warning(f"Failed to initialize enhanced mode: {e}")
        return run_flow(data)  # Fallback to basic workflow

    # Use enhanced workflow
    return run_flow(data)


def get_workflow_status(session_id, memory):
    """
    Get detailed workflow status for a session
    """
    if not memory:
        return {"error": "No memory available"}

    workflow_logs = memory.get(session_id)

    if not workflow_logs:
        return {"error": "No workflow logs found"}

    status = {
        'session_id': session_id,
        'total_events': len(workflow_logs),
        'workflow_events': [],
        'current_status': 'unknown',
        'has_errors': False,
        'completion_status': 'unknown'
    }

    for log_entry in workflow_logs:
        event = log_entry.get('event', '')
        data = log_entry.get('data', {})

        if 'workflow_start' in event:
            status['current_status'] = 'started'
        elif 'workflow_complete' in event:
            status['current_status'] = 'completed'
            status['completion_status'] = 'success' if not data.get('has_error') else 'error'
        elif 'error' in event.lower():
            status['has_errors'] = True

        status['workflow_events'].append({
            'event': event,
            'data': data
        })

    return status


# Backward compatibility function
def run_basic_flow(data):
    """
    Run basic workflow (alias for backward compatibility)
    """
    return run_flow(data)


# Utility functions for workflow management
def validate_workflow_data(data):
    """
    Validate data before running workflow
    """
    if data is None:
        return False, "No data provided"

    if hasattr(data, 'empty') and data.empty:
        return False, "Data is empty"

    if hasattr(data, 'shape') and data.shape[0] == 0:
        return False, "Data has no rows"

    # Check for required columns
    required_columns = ['Account_ID']
    if hasattr(data, 'columns'):
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"

    return True, "Data validation passed"


def create_workflow_summary(state, memory):
    """
    Create a summary of the workflow execution
    """
    summary = {
        'session_id': state.session_id,
        'final_step': state.step,
        'has_error': bool(state.error),
        'error_message': state.error if state.error else None,
        'has_result': bool(state.result),
        'data_processed': True,
        'enhanced_features_used': hasattr(state, 'dormant_summary_report'),
        'analysis_mode': 'unknown'
    }

    # Add analysis mode if available
    if hasattr(state, 'dormant_results'):
        summary['dormancy_analysis_completed'] = True
        if hasattr(state, 'dormant_summary_report'):
            summary['analysis_mode'] = 'enhanced'
        else:
            summary['analysis_mode'] = 'legacy'

    # Add compliance information
    if hasattr(state, 'compliance_results'):
        summary['compliance_analysis_completed'] = True

    # Add memory statistics
    if memory:
        workflow_logs = memory.get(state.session_id)
        summary['total_log_entries'] = len(workflow_logs) if workflow_logs else 0

    return summary