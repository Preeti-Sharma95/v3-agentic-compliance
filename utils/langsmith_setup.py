import os
import logging
from typing import Optional
from config.config import config

logger = logging.getLogger(__name__)


def setup_langsmith() -> bool:
    """Setup LangSmith tracing"""
    try:
        if not config.langsmith_enabled:
            logger.info("LangSmith tracing disabled")
            return False

        if not config.langsmith_api_key:
            logger.warning("LangSmith API key not provided")
            return False

        # Set environment variables for LangSmith
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = config.langsmith_project
        os.environ["LANGCHAIN_ENDPOINT"] = config.langsmith_endpoint
        os.environ["LANGCHAIN_API_KEY"] = config.langsmith_api_key

        logger.info(f"LangSmith tracing enabled for project: {config.langsmith_project}")
        return True

    except Exception as e:
        logger.error(f"Failed to setup LangSmith: {e}")
        return False


def get_trace_url(session_id: str) -> Optional[str]:
    """Get LangSmith trace URL for a session"""
    if not config.langsmith_enabled:
        return None

    base_url = config.langsmith_endpoint.replace("api.", "")
    return f"{base_url}/o/{config.langsmith_project}/sessions/{session_id}"