"""
Configuration management for Linux AI Agent.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class ScannerConfig:
    """Scanner configuration settings."""
    entropy_threshold: float = 7.5
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    clamav_timeout: int = 30
    heuristic_enabled: bool = True
    quarantine_enabled: bool = True


@dataclass
class AIConfig:
    """AI configuration settings."""
    intent_confidence_threshold: float = 0.5
    rl_learning_rate: float = 0.001
    rl_epsilon_decay: float = 0.995
    rl_epsilon_min: float = 0.01
    gemma_enabled: bool = False
    enhanced_nlp_enabled: bool = False


@dataclass
class MonitorConfig:
    """Monitoring configuration settings."""
    disk_warning_threshold: float = 80.0  # percentage
    disk_critical_threshold: float = 95.0  # percentage
    monitor_interval: int = 60  # seconds
    log_retention_days: int = 30


@dataclass
class AppConfig:
    """Main application configuration."""
    debug: bool = False
    log_level: str = "INFO"
    log_file: str = "logs/agent.log"
    quarantine_dir: str = "quarantine"
    data_dir: str = "data"
    scanner: ScannerConfig = None
    ai: AIConfig = None
    monitor: MonitorConfig = None

    def __post_init__(self):
        if self.scanner is None:
            self.scanner = ScannerConfig()
        if self.ai is None:
            self.ai = AIConfig()
        if self.monitor is None:
            self.monitor = MonitorConfig()


class ConfigManager:
    """Configuration manager for the Linux AI Agent."""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = Path(config_file)
        self._config = None
        self.load_config()
    
    @property
    def config(self) -> AppConfig:
        """Get the current configuration."""
        if self._config is None:
            self._config = AppConfig()
        return self._config
    
    def load_config(self) -> AppConfig:
        """Load configuration from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                
                # Convert nested dicts to dataclasses
                scanner_data = data.get('scanner', {})
                ai_data = data.get('ai', {})
                monitor_data = data.get('monitor', {})
                
                scanner_config = ScannerConfig(**scanner_data)
                ai_config = AIConfig(**ai_data)
                monitor_config = MonitorConfig(**monitor_data)
                
                # Remove nested configs from main data
                config_data = {k: v for k, v in data.items() 
                             if k not in ['scanner', 'ai', 'monitor']}
                
                self._config = AppConfig(
                    scanner=scanner_config,
                    ai=ai_config,
                    monitor=monitor_config,
                    **config_data
                )
                
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                print(f"⚠️ Error loading config: {e}")
                print("Using default configuration")
                self._config = AppConfig()
        else:
            self._config = AppConfig()
            self.save_config()  # Create default config file
        
        return self._config
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        try:
            # Create directory if it doesn't exist
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to dict for JSON serialization
            config_dict = asdict(self._config)
            
            with open(self.config_file, 'w') as f:
                json.dump(config_dict, f, indent=4)
                
        except Exception as e:
            print(f"❌ Error saving config: {e}")
    
    def update_config(self, **kwargs) -> None:
        """Update configuration values."""
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
        self.save_config()
    
    def get_env_config(self) -> Dict[str, Any]:
        """Get configuration from environment variables."""
        env_config = {}
        
        # Check for environment variable overrides
        if os.getenv('LAI_DEBUG'):
            env_config['debug'] = os.getenv('LAI_DEBUG').lower() == 'true'
        
        if os.getenv('LAI_LOG_LEVEL'):
            env_config['log_level'] = os.getenv('LAI_LOG_LEVEL')
        
        if os.getenv('LAI_QUARANTINE_DIR'):
            env_config['quarantine_dir'] = os.getenv('LAI_QUARANTINE_DIR')
        
        # Scanner settings
        if os.getenv('LAI_ENTROPY_THRESHOLD'):
            try:
                threshold = float(os.getenv('LAI_ENTROPY_THRESHOLD'))
                if not hasattr(env_config, 'scanner'):
                    env_config['scanner'] = {}
                env_config['scanner']['entropy_threshold'] = threshold
            except ValueError:
                pass
        
        return env_config


# Global configuration instance
_config_manager = None

def get_config() -> AppConfig:
    """Get the global configuration instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager.config

def reload_config() -> AppConfig:
    """Reload configuration from file."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    else:
        _config_manager.load_config()
    return _config_manager.config