# config/memory_config.py
"""
Configuration for Hybrid Memory Agent and MCP Integration
"""

import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """Memory system configuration"""
    # Cache settings
    session_cache_size: int = 500
    knowledge_cache_size: int = 1000
    cache_ttl_hours: int = 24

    # Persistence settings
    enable_persistence: bool = True
    persistence_path: str = "data/memory"
    auto_backup_interval_hours: int = 6

    # Performance settings
    cleanup_interval_hours: int = 4
    max_memory_size_mb: int = 512
    compression_enabled: bool = True


@dataclass
class MCPConfig:
    """MCP integration configuration"""
    # Connection settings
    server_url: str = "http://localhost:8000"
    timeout_seconds: int = 30
    max_retries: int = 3
    connection_pool_size: int = 10

    # Feature flags
    enabled: bool = False
    knowledge_base_enabled: bool = True
    regulatory_guidance_enabled: bool = True
    compliance_validation_enabled: bool = True
    batch_processing_enabled: bool = True

    # Performance settings
    request_rate_limit: int = 100  # requests per minute
    batch_size: int = 50
    parallel_requests: int = 5

    # Caching
    enable_response_caching: bool = True
    cache_duration_hours: int = 24


@dataclass
class EnhancedConfig:
    """Enhanced system configuration"""
    # System settings
    environment: str = "development"
    debug_mode: bool = True
    log_level: str = "INFO"

    # Feature flags
    enhanced_mode_enabled: bool = True
    memory_system_enabled: bool = True
    mcp_integration_enabled: bool = False
    langsmith_enabled: bool = False

    # Performance
    async_processing: bool = True
    parallel_agents: bool = True
    max_concurrent_sessions: int = 10

    # Security
    enable_encryption: bool = False
    api_key_required: bool = False
    session_timeout_hours: int = 24

    # Sub-configurations
    memory: MemoryConfig = None
    mcp: MCPConfig = None

    def __post_init__(self):
        if self.memory is None:
            self.memory = MemoryConfig()
        if self.mcp is None:
            self.mcp = MCPConfig()


class ConfigManager:
    """Configuration manager for the enhanced system"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.getenv('CONFIG_PATH', 'config/enhanced_config.json')
        self._config: Optional[EnhancedConfig] = None
        self._load_config()

    def _load_config(self):
        """Load configuration from environment and files"""
        # Start with defaults
        self._config = EnhancedConfig()

        # Override with environment variables
        self._load_from_environment()

        # Override with config file if exists
        if os.path.exists(self.config_path):
            self._load_from_file()

        # Validate configuration
        self._validate_config()

    def _load_from_environment(self):
        """Load configuration from environment variables"""
        # System settings
        self._config.environment = os.getenv('ENVIRONMENT', self._config.environment)
        self._config.debug_mode = os.getenv('DEBUG_MODE', 'true').lower() == 'true'
        self._config.log_level = os.getenv('LOG_LEVEL', self._config.log_level)

        # Feature flags
        self._config.enhanced_mode_enabled = os.getenv('ENHANCED_MODE_ENABLED', 'true').lower() == 'true'
        self._config.memory_system_enabled = os.getenv('MEMORY_SYSTEM_ENABLED', 'true').lower() == 'true'
        self._config.mcp_integration_enabled = os.getenv('MCP_INTEGRATION_ENABLED', 'false').lower() == 'true'
        self._config.langsmith_enabled = os.getenv('LANGSMITH_ENABLED', 'false').lower() == 'true'

        # Memory configuration
        self._config.memory.session_cache_size = int(os.getenv('MEMORY_SESSION_CACHE_SIZE',
                                                               self._config.memory.session_cache_size))
        self._config.memory.knowledge_cache_size = int(os.getenv('MEMORY_KNOWLEDGE_CACHE_SIZE',
                                                                 self._config.memory.knowledge_cache_size))
        self._config.memory.enable_persistence = os.getenv('MEMORY_PERSISTENCE_ENABLED', 'true').lower() == 'true'
        self._config.memory.persistence_path = os.getenv('MEMORY_PERSISTENCE_PATH',
                                                         self._config.memory.persistence_path)

        # MCP configuration
        self._config.mcp.server_url = os.getenv('MCP_SERVER_URL', self._config.mcp.server_url)
        self._config.mcp.enabled = os.getenv('MCP_ENABLED', 'false').lower() == 'true'
        self._config.mcp.timeout_seconds = int(os.getenv('MCP_TIMEOUT_SECONDS', self._config.mcp.timeout_seconds))
        self._config.mcp.max_retries = int(os.getenv('MCP_MAX_RETRIES', self._config.mcp.max_retries))
        self._config.mcp.batch_size = int(os.getenv('MCP_BATCH_SIZE', self._config.mcp.batch_size))

        # LangSmith configuration
        if self._config.langsmith_enabled:
            self.langsmith_project = os.getenv('LANGCHAIN_PROJECT', 'banking-compliance')
            self.langsmith_endpoint = os.getenv('LANGCHAIN_ENDPOINT', 'https://api.smith.langchain.com')
            self.langsmith_api_key = os.getenv('LANGCHAIN_API_KEY')

    def _load_from_file(self):
        """Load configuration from JSON file"""
        try:
            import json
            with open(self.config_path, 'r') as f:
                file_config = json.load(f)

            # Update configuration with file values
            self._update_config_from_dict(file_config)

        except Exception as e:
            logger.warning(f"Failed to load config file {self.config_path}: {e}")

    def _update_config_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary"""
        for key, value in config_dict.items():
            if hasattr(self._config, key):
                if isinstance(value, dict) and hasattr(getattr(self._config, key), '__dict__'):
                    # Handle nested configuration objects
                    nested_config = getattr(self._config, key)
                    for nested_key, nested_value in value.items():
                        if hasattr(nested_config, nested_key):
                            setattr(nested_config, nested_key, nested_value)
                else:
                    setattr(self._config, key, value)

    def _validate_config(self):
        """Validate configuration settings"""
        # Validate memory settings
        if self._config.memory.session_cache_size <= 0:
            raise ValueError("Session cache size must be positive")

        if self._config.memory.knowledge_cache_size <= 0:
            raise ValueError("Knowledge cache size must be positive")

        # Validate MCP settings
        if self._config.mcp.enabled:
            if not self._config.mcp.server_url:
                raise ValueError("MCP server URL is required when MCP is enabled")

            if self._config.mcp.timeout_seconds <= 0:
                raise ValueError("MCP timeout must be positive")

        # Create persistence directory if needed
        if self._config.memory.enable_persistence:
            persistence_path = Path(self._config.memory.persistence_path)
            persistence_path.mkdir(parents=True, exist_ok=True)

    @property
    def config(self) -> EnhancedConfig:
        """Get the current configuration"""
        return self._config

    def get_memory_config(self) -> MemoryConfig:
        """Get memory configuration"""
        return self._config.memory

    def get_mcp_config(self) -> MCPConfig:
        """Get MCP configuration"""
        return self._config.mcp

    def is_enhanced_mode(self) -> bool:
        """Check if enhanced mode is enabled"""
        return self._config.enhanced_mode_enabled

    def is_memory_enabled(self) -> bool:
        """Check if memory system is enabled"""
        return self._config.memory_system_enabled

    def is_mcp_enabled(self) -> bool:
        """Check if MCP integration is enabled"""
        return self._config.mcp_integration_enabled and self._config.mcp.enabled

    def save_config(self, path: Optional[str] = None):
        """Save current configuration to file"""
        save_path = path or self.config_path

        try:
            import json
            from dataclasses import asdict

            config_dict = asdict(self._config)

            # Create directory if needed
            save_dir = Path(save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)

            with open(save_path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)

            logger.info(f"Configuration saved to {save_path}")

        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")

    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for display"""
        return {
            'system': {
                'environment': self._config.environment,
                'enhanced_mode': self._config.enhanced_mode_enabled,
                'debug_mode': self._config.debug_mode
            },
            'memory': {
                'enabled': self._config.memory_system_enabled,
                'session_cache_size': self._config.memory.session_cache_size,
                'knowledge_cache_size': self._config.memory.knowledge_cache_size,
                'persistence_enabled': self._config.memory.enable_persistence
            },
            'mcp': {
                'enabled': self.is_mcp_enabled(),
                'server_url': self._config.mcp.server_url,
                'timeout': self._config.mcp.timeout_seconds,
                'batch_size': self._config.mcp.batch_size
            },
            'langsmith': {
                'enabled': self._config.langsmith_enabled,
                'project': getattr(self, 'langsmith_project', 'N/A')
            }
        }


# Global configuration instance
config_manager = ConfigManager()
enhanced_config = config_manager.config


# Environment-specific configurations
class DevelopmentConfig(EnhancedConfig):
    """Development environment configuration"""

    def __init__(self):
        super().__init__()
        self.environment = "development"
        self.debug_mode = True
        self.log_level = "DEBUG"
        self.mcp.enabled = False  # Disable MCP in development by default


class ProductionConfig(EnhancedConfig):
    """Production environment configuration"""

    def __init__(self):
        super().__init__()
        self.environment = "production"
        self.debug_mode = False
        self.log_level = "INFO"
        self.memory.enable_persistence = True
        self.memory.auto_backup_interval_hours = 2
        self.enable_encryption = True
        self.api_key_required = True


class TestingConfig(EnhancedConfig):
    """Testing environment configuration"""

    def __init__(self):
        super().__init__()
        self.environment = "testing"
        self.debug_mode = True
        self.log_level = "DEBUG"
        self.memory.session_cache_size = 50
        self.memory.knowledge_cache_size = 100
        self.memory.enable_persistence = False
        self.mcp.enabled = False


def get_config_for_environment(env: str) -> EnhancedConfig:
    """Get configuration for specific environment"""
    configs = {
        'development': DevelopmentConfig,
        'production': ProductionConfig,
        'testing': TestingConfig
    }

    config_class = configs.get(env, DevelopmentConfig)
    return config_class()


# Configuration validation utilities
def validate_mcp_connection(config: MCPConfig) -> bool:
    """Validate MCP connection configuration"""
    if not config.enabled:
        return True

    try:
        import requests
        response = requests.get(f"{config.server_url}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def validate_memory_paths(config: MemoryConfig) -> bool:
    """Validate memory persistence paths"""
    if not config.enable_persistence:
        return True

    try:
        persistence_path = Path(config.persistence_path)
        persistence_path.mkdir(parents=True, exist_ok=True)

        # Test write permissions
        test_file = persistence_path / "test_write.tmp"
        test_file.write_text("test")
        test_file.unlink()

        return True
    except Exception:
        return False


# Configuration factory
class ConfigFactory:
    """Factory for creating configurations"""

    @staticmethod
    def create_config(
            environment: str = "development",
            memory_enabled: bool = True,
            mcp_enabled: bool = False,
            **kwargs
    ) -> EnhancedConfig:
        """Create configuration with specified parameters"""

        config = get_config_for_environment(environment)
        config.memory_system_enabled = memory_enabled
        config.mcp_integration_enabled = mcp_enabled
        config.mcp.enabled = mcp_enabled

        # Apply additional parameters
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            elif '.' in key:
                # Handle nested attributes like 'memory.cache_size'
                parts = key.split('.')
                obj = config
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)

        return config

    @staticmethod
    def create_memory_config(**kwargs) -> MemoryConfig:
        """Create memory configuration with specified parameters"""
        config = MemoryConfig()

        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config

    @staticmethod
    def create_mcp_config(**kwargs) -> MCPConfig:
        """Create MCP configuration with specified parameters"""
        config = MCPConfig()

        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config


# Configuration loading utilities
def load_config_from_yaml(yaml_path: str) -> EnhancedConfig:
    """Load configuration from YAML file"""
    try:
        import yaml

        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)

        config = EnhancedConfig()
        config_manager = ConfigManager()
        config_manager._update_config_from_dict(yaml_data)

        return config_manager.config

    except Exception as e:
        raise ValueError(f"Failed to load YAML config: {e}")


def export_config_to_yaml(config: EnhancedConfig, yaml_path: str):
    """Export configuration to YAML file"""
    try:
        import yaml
        from dataclasses import asdict

        config_dict = asdict(config)

        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

        logger.info(f"Configuration exported to {yaml_path}")

    except Exception as e:
        raise ValueError(f"Failed to export YAML config: {e}")


# Monitoring and health check utilities
class ConfigMonitor:
    """Monitor configuration health and performance"""

    def __init__(self, config: EnhancedConfig):
        self.config = config
        self.health_status = {}

    async def check_system_health(self) -> Dict[str, bool]:
        """Check overall system health"""
        health = {}

        # Check memory system
        if self.config.memory_system_enabled:
            health['memory_system'] = validate_memory_paths(self.config.memory)

        # Check MCP connection
        if self.config.mcp_integration_enabled:
            health['mcp_connection'] = validate_mcp_connection(self.config.mcp)

        # Check disk space for persistence
        if self.config.memory.enable_persistence:
            health['disk_space'] = self._check_disk_space()

        self.health_status = health
        return health

    def _check_disk_space(self) -> bool:
        """Check available disk space for persistence"""
        try:
            import shutil

            persistence_path = Path(self.config.memory.persistence_path)
            _, _, free_bytes = shutil.disk_usage(persistence_path.parent)
            free_gb = free_bytes / (1024 ** 3)

            # Require at least 1GB free space
            return free_gb >= 1.0

        except Exception:
            return False

    def get_health_report(self) -> Dict[str, Any]:
        """Get detailed health report"""
        return {
            'timestamp': datetime.now().isoformat(),
            'system_health': self.health_status,
            'configuration_summary': config_manager.get_config_summary(),
            'recommendations': self._generate_recommendations()
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate configuration recommendations"""
        recommendations = []

        if not self.health_status.get('memory_system', True):
            recommendations.append("Fix memory system persistence path issues")

        if not self.health_status.get('mcp_connection', True):
            recommendations.append("Check MCP server connectivity")

        if not self.health_status.get('disk_space', True):
            recommendations.append("Free up disk space for memory persistence")

        return recommendations


# Example usage and testing
if __name__ == "__main__":
    # Example configuration usage
    print("=== Enhanced Banking Compliance Configuration ===")

    # Load default configuration
    config = config_manager.config
    print(f"Environment: {config.environment}")
    print(f"Enhanced Mode: {config.enhanced_mode_enabled}")
    print(f"Memory System: {config.memory_system_enabled}")
    print(f"MCP Integration: {config_manager.is_mcp_enabled()}")

    # Display configuration summary
    summary = config_manager.get_config_summary()
    print("\nConfiguration Summary:")
    for section, settings in summary.items():
        print(f"  {section.upper()}:")
        for key, value in settings.items():
            print(f"    {key}: {value}")

    # Test configuration factory
    print("\n=== Testing Configuration Factory ===")
    test_config = ConfigFactory.create_config(
        environment="production",
        memory_enabled=True,
        mcp_enabled=True
    )
    print(f"Test config environment: {test_config.environment}")
    print(f"Test config MCP enabled: {test_config.mcp.enabled}")

    # Test health monitoring
    print("\n=== Testing Health Monitor ===")
    monitor = ConfigMonitor(config)
    import asyncio


    async def test_health():
        health = await monitor.check_system_health()
        print("Health status:", health)

        report = monitor.get_health_report()
        print("Health report generated")


    asyncio.run(test_health())