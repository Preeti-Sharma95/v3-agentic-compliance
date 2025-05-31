# config/config.py
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class DormancyConfig:
    """Configuration for dormancy analysis thresholds"""
    standard_inactivity_years: int = int(os.getenv("STANDARD_INACTIVITY_YEARS", "3"))
    payment_instrument_unclaimed_years: int = int(os.getenv("PAYMENT_INSTRUMENT_UNCLAIMED_YEARS", "1"))
    sdb_unpaid_fees_years: int = int(os.getenv("SDB_UNPAID_FEES_YEARS", "3"))
    eligibility_for_cb_transfer_years: int = int(os.getenv("ELIGIBILITY_FOR_CB_TRANSFER_YEARS", "5"))
    high_value_threshold: float = float(os.getenv("HIGH_VALUE_THRESHOLD", "25000"))
    warning_period_years: float = float(os.getenv("WARNING_PERIOD_YEARS", "2.5"))


@dataclass
class Config:
    """Configuration management for the enhanced compliance system"""

    # LangSmith Configuration
    langsmith_api_key: str = os.getenv("LANGSMITH_API_KEY", "")
    langsmith_project: str = os.getenv("LANGSMITH_PROJECT", "banking-compliance")
    langsmith_endpoint: str = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    langsmith_enabled: bool = bool(os.getenv("LANGSMITH_ENABLED", "true").lower() == "true")

    # Anthropic Configuration
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")

    # MCP Configuration
    mcp_server_url: str = os.getenv("MCP_SERVER_URL", "http://localhost:8000")
    mcp_enabled: bool = bool(os.getenv("MCP_ENABLED", "true").lower() == "true")

    # CBUAE Compliance Configuration
    standard_inactivity_years: int = int(os.getenv("STANDARD_INACTIVITY_YEARS", "3"))
    payment_instrument_unclaimed_years: int = int(os.getenv("PAYMENT_INSTRUMENT_UNCLAIMED_YEARS", "1"))
    sdb_unpaid_fees_years: int = int(os.getenv("SDB_UNPAID_FEES_YEARS", "3"))
    eligibility_for_cb_transfer_years: int = int(os.getenv("ELIGIBILITY_FOR_CB_TRANSFER_YEARS", "5"))

    # Application Configuration
    environment: str = os.getenv("ENVIRONMENT", "development")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # Enhanced Dormancy Configuration
    enhanced_dormancy_enabled: bool = bool(os.getenv("ENHANCED_DORMANCY_ENABLED", "true").lower() == "true")
    dormancy_config: DormancyConfig = None

    def __post_init__(self):
        if self.dormancy_config is None:
            self.dormancy_config = DormancyConfig()

    def validate(self) -> bool:
        """Validate critical configuration"""
        if self.langsmith_enabled and not self.langsmith_api_key:
            print("Warning: LangSmith enabled but no API key provided")
            return False

        if self.mcp_enabled and not self.mcp_server_url:
            print("Warning: MCP enabled but no server URL provided")
            return False

        return True

    def get_dormancy_thresholds(self) -> dict:
        """Get dormancy analysis thresholds"""
        return {
            'standard_inactivity_years': self.dormancy_config.standard_inactivity_years,
            'payment_instrument_unclaimed_years': self.dormancy_config.payment_instrument_unclaimed_years,
            'sdb_unpaid_fees_years': self.dormancy_config.sdb_unpaid_fees_years,
            'eligibility_for_cb_transfer_years': self.dormancy_config.eligibility_for_cb_transfer_years,
            'high_value_threshold': self.dormancy_config.high_value_threshold,
            'warning_period_years': self.dormancy_config.warning_period_years
        }


# Global config instance
config = Config()