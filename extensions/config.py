"""
Configuration Management Module

This module handles loading and validation of extension-specific configurations.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import yaml
from pathlib import Path


@dataclass
class EnhancedRewardConfig:
    """Configuration for enhanced reward components."""
    enable_enhanced: bool = True
    schema_weight: float = -0.5
    structural_select_weight: float = 0.3
    structural_where_weight: float = 0.3
    structural_join_weight: float = 0.2
    syntax_weight: float = 0.2


@dataclass
class OptimizationConfig:
    """Configuration for 24GB GPU optimization."""
    model_name: str = "Qwen/Qwen2.5-Coder-3B-Instruct"
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"
    max_prompt_length: int = 4096
    max_response_length: int = 2048


@dataclass
class LoggingConfig:
    """Configuration for enhanced logging."""
    log_interval: int = 10
    save_interval: int = 500
    eval_interval: int = 100
    detailed_rewards: bool = True
    track_gpu_memory: bool = True


@dataclass
class ExtensionConfig:
    """Main configuration for all extensions."""
    reward: EnhancedRewardConfig = field(default_factory=EnhancedRewardConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(config_path: str) -> ExtensionConfig:
    """
    Load extension configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        ExtensionConfig object
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        print(f"[Config] Warning: Config file {config_path} not found, using defaults")
        return ExtensionConfig()
    
    try:
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        if not config_dict:
            print("[Config] Warning: Empty config file, using defaults")
            return ExtensionConfig()
        
        # Parse reward config
        reward_config = EnhancedRewardConfig()
        if 'reward' in config_dict:
            reward_dict = config_dict['reward']
            reward_config = EnhancedRewardConfig(
                enable_enhanced=reward_dict.get('enable_enhanced', True),
                schema_weight=reward_dict.get('schema_weight', -0.5),
                structural_select_weight=reward_dict.get('structural_select_weight', 0.3),
                structural_where_weight=reward_dict.get('structural_where_weight', 0.3),
                structural_join_weight=reward_dict.get('structural_join_weight', 0.2),
                syntax_weight=reward_dict.get('syntax_weight', 0.2)
            )
        
        # Parse optimization config
        opt_config = OptimizationConfig()
        if 'optimization' in config_dict:
            opt_dict = config_dict['optimization']
            opt_config = OptimizationConfig(
                model_name=opt_dict.get('model_name', "Qwen/Qwen2.5-Coder-3B-Instruct"),
                batch_size=opt_dict.get('batch_size', 2),
                gradient_accumulation_steps=opt_dict.get('gradient_accumulation_steps', 8),
                gradient_checkpointing=opt_dict.get('gradient_checkpointing', True),
                mixed_precision=opt_dict.get('mixed_precision', 'bf16'),
                max_prompt_length=opt_dict.get('max_prompt_length', 4096),
                max_response_length=opt_dict.get('max_response_length', 2048)
            )
        
        # Parse logging config
        log_config = LoggingConfig()
        if 'logging' in config_dict:
            log_dict = config_dict['logging']
            log_config = LoggingConfig(
                log_interval=log_dict.get('log_interval', 10),
                save_interval=log_dict.get('save_interval', 500),
                eval_interval=log_dict.get('eval_interval', 100),
                detailed_rewards=log_dict.get('detailed_rewards', True),
                track_gpu_memory=log_dict.get('track_gpu_memory', True)
            )
        
        config = ExtensionConfig(
            reward=reward_config,
            optimization=opt_config,
            logging=log_config
        )
        
        validate_config(config)
        return config
        
    except Exception as e:
        print(f"[Config] Error loading config: {e}, using defaults")
        return ExtensionConfig()


def validate_config(config: ExtensionConfig) -> bool:
    """
    Validate configuration values.
    
    Args:
        config: Configuration to validate
    
    Returns:
        True if valid, raises ValueError otherwise
    """
    # Validate reward weights
    if config.reward.schema_weight > 0:
        print("[Config] Warning: schema_weight should be negative (penalty)")
    
    if config.reward.structural_select_weight < 0:
        raise ValueError("structural_select_weight must be non-negative")
    
    if config.reward.structural_where_weight < 0:
        raise ValueError("structural_where_weight must be non-negative")
    
    if config.reward.structural_join_weight < 0:
        raise ValueError("structural_join_weight must be non-negative")
    
    if config.reward.syntax_weight < 0:
        raise ValueError("syntax_weight must be non-negative")
    
    # Validate optimization settings
    if config.optimization.batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    
    if config.optimization.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    
    if config.optimization.mixed_precision not in ['no', 'fp16', 'bf16']:
        raise ValueError("mixed_precision must be 'no', 'fp16', or 'bf16'")
    
    # Validate logging settings
    if config.logging.log_interval < 1:
        raise ValueError("log_interval must be at least 1")
    
    if config.logging.save_interval < 1:
        raise ValueError("save_interval must be at least 1")
    
    if config.logging.eval_interval < 1:
        raise ValueError("eval_interval must be at least 1")
    
    return True
