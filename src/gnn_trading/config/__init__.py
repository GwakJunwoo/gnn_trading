"""
gnn_trading.config.manager
==========================

Configuration management system for production deployment
Handles environment-specific configurations, secrets, and validation
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import yaml
import json
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    name: str = "gnn_trading"
    user: str = "gnn_user"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False


@dataclass
class RedisConfig:
    """Redis configuration"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    max_connections: int = 50
    decode_responses: bool = True


@dataclass
class APIConfig:
    """API server configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    debug: bool = False
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    max_request_size: int = 16 * 1024 * 1024  # 16MB
    rate_limit: int = 1000  # requests per minute


@dataclass
class ModelConfig:
    """Model configuration"""
    model_dir: str = "models"
    ensemble_enabled: bool = True
    max_ensemble_models: int = 5
    auto_reload: bool = False
    device: str = "auto"  # auto, cpu, cuda
    batch_size: int = 32
    max_sequence_length: int = 1000


@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration"""
    enabled: bool = True
    metrics_port: int = 9090
    health_check_interval: int = 60
    alert_email_enabled: bool = False
    alert_slack_enabled: bool = False
    prometheus_enabled: bool = True
    grafana_enabled: bool = True


@dataclass
class SecurityConfig:
    """Security configuration"""
    secret_key: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440  # 24 hours
    api_key_required: bool = False
    rate_limiting_enabled: bool = True
    cors_enabled: bool = True
    https_only: bool = False


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_enabled: bool = True
    file_path: str = "logs/gnn_trading.log"
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    backup_count: int = 5
    console_enabled: bool = True


@dataclass
class AppConfig:
    """Main application configuration"""
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    testing: bool = False
    
    # Component configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    api: APIConfig = field(default_factory=APIConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Additional settings
    data_dir: str = "data"
    temp_dir: str = "tmp"
    max_workers: int = 4
    timezone: str = "UTC"


class ConfigManager:
    """Configuration manager with environment support"""
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        self.config_dir = Path(config_dir) if config_dir else Path("configs")
        self.environment = self._detect_environment()
        self._config: Optional[AppConfig] = None
        
    def _detect_environment(self) -> Environment:
        """Detect current environment from environment variables"""
        env_name = os.getenv("ENV", os.getenv("ENVIRONMENT", "development")).lower()
        
        try:
            return Environment(env_name)
        except ValueError:
            logger.warning(f"Unknown environment '{env_name}', defaulting to development")
            return Environment.DEVELOPMENT
            
    def load_config(self, config_file: Optional[str] = None) -> AppConfig:
        """Load configuration from file and environment variables"""
        if config_file is None:
            config_file = f"{self.environment.value}_config.yaml"
            
        config_path = self.config_dir / config_file
        
        # Start with default configuration
        config_dict = {}
        
        # Load from file if exists
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f) or {}
                
        # Override with environment variables
        config_dict = self._apply_env_overrides(config_dict)
        
        # Create configuration object
        self._config = self._dict_to_config(config_dict)
        
        # Validate configuration
        self._validate_config(self._config)
        
        logger.info(f"Configuration loaded for {self.environment.value} environment")
        return self._config
        
    def _apply_env_overrides(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides"""
        # Define environment variable mappings
        env_mappings = {
            # Database
            "POSTGRES_HOST": ["database", "host"],
            "POSTGRES_PORT": ["database", "port"],
            "POSTGRES_DB": ["database", "name"],
            "POSTGRES_USER": ["database", "user"],
            "POSTGRES_PASSWORD": ["database", "password"],
            
            # Redis
            "REDIS_HOST": ["redis", "host"],
            "REDIS_PORT": ["redis", "port"],
            "REDIS_PASSWORD": ["redis", "password"],
            
            # API
            "API_HOST": ["api", "host"],
            "API_PORT": ["api", "port"],
            "API_WORKERS": ["api", "workers"],
            "API_DEBUG": ["api", "debug"],
            
            # Security
            "SECRET_KEY": ["security", "secret_key"],
            "JWT_SECRET": ["security", "secret_key"],
            "API_KEY_REQUIRED": ["security", "api_key_required"],
            
            # Model
            "MODEL_DIR": ["model", "model_dir"],
            "MODEL_DEVICE": ["model", "device"],
            
            # Monitoring
            "MONITORING_ENABLED": ["monitoring", "enabled"],
            
            # Logging
            "LOG_LEVEL": ["logging", "level"],
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Navigate to nested dictionary location
                current = config_dict
                for key in config_path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                
                # Convert value to appropriate type
                final_key = config_path[-1]
                current[final_key] = self._convert_env_value(env_value)
                
        return config_dict
        
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type"""
        # Boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
            
        # Integer conversion
        try:
            return int(value)
        except ValueError:
            pass
            
        # Float conversion
        try:
            return float(value)
        except ValueError:
            pass
            
        # JSON conversion for complex types
        if value.startswith(('[', '{')):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
                
        # Return as string
        return value
        
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> AppConfig:
        """Convert dictionary to AppConfig object"""
        def create_dataclass_from_dict(cls, data):
            if not isinstance(data, dict):
                return data
                
            # Get field names and types
            field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
            kwargs = {}
            
            for field_name, field_type in field_types.items():
                if field_name in data:
                    value = data[field_name]
                    
                    # Handle nested dataclasses
                    if hasattr(field_type, '__dataclass_fields__'):
                        kwargs[field_name] = create_dataclass_from_dict(field_type, value)
                    else:
                        kwargs[field_name] = value
                        
            return cls(**kwargs)
            
        return create_dataclass_from_dict(AppConfig, config_dict)
        
    def _validate_config(self, config: AppConfig):
        """Validate configuration values"""
        errors = []
        
        # Validate database configuration
        if not config.database.host:
            errors.append("Database host is required")
        if not (1 <= config.database.port <= 65535):
            errors.append("Database port must be between 1 and 65535")
            
        # Validate API configuration
        if not (1 <= config.api.port <= 65535):
            errors.append("API port must be between 1 and 65535")
        if config.api.workers < 1:
            errors.append("API workers must be at least 1")
            
        # Validate security configuration
        if config.environment == Environment.PRODUCTION:
            if not config.security.secret_key:
                errors.append("Secret key is required in production")
            if len(config.security.secret_key) < 32:
                errors.append("Secret key must be at least 32 characters in production")
                
        # Validate model configuration
        if not Path(config.model.model_dir).exists():
            Path(config.model.model_dir).mkdir(parents=True, exist_ok=True)
            
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
            
    def get_config(self) -> AppConfig:
        """Get current configuration"""
        if self._config is None:
            self.load_config()
        return self._config
        
    def save_config(self, config_file: Optional[str] = None):
        """Save current configuration to file"""
        if self._config is None:
            raise ValueError("No configuration loaded")
            
        if config_file is None:
            config_file = f"{self.environment.value}_config.yaml"
            
        config_path = self.config_dir / config_file
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert config to dictionary
        config_dict = self._config_to_dict(self._config)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
    def _config_to_dict(self, config: Any) -> Dict[str, Any]:
        """Convert configuration object to dictionary"""
        if hasattr(config, '__dataclass_fields__'):
            result = {}
            for field_name in config.__dataclass_fields__:
                value = getattr(config, field_name)
                if hasattr(value, '__dataclass_fields__'):
                    result[field_name] = self._config_to_dict(value)
                elif isinstance(value, Enum):
                    result[field_name] = value.value
                else:
                    result[field_name] = value
            return result
        return config
        
    def update_config(self, **kwargs):
        """Update configuration values"""
        if self._config is None:
            self.load_config()
            
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
            else:
                logger.warning(f"Unknown configuration key: {key}")
                
    def get_database_url(self) -> str:
        """Get database connection URL"""
        db = self._config.database
        return f"postgresql://{db.user}:{db.password}@{db.host}:{db.port}/{db.name}"
        
    def get_redis_url(self) -> str:
        """Get Redis connection URL"""
        redis = self._config.redis
        password_part = f":{redis.password}@" if redis.password else ""
        return f"redis://{password_part}{redis.host}:{redis.port}/{redis.db}"


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config() -> AppConfig:
    """Get application configuration"""
    return get_config_manager().get_config()


def load_config(config_dir: Optional[str] = None, config_file: Optional[str] = None) -> AppConfig:
    """Load application configuration"""
    global _config_manager
    if config_dir:
        _config_manager = ConfigManager(config_dir)
    else:
        _config_manager = get_config_manager()
    return _config_manager.load_config(config_file)
