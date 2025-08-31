"""
Configuration loading utilities
"""
import yaml
import os
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_all_configs(config_dir: str = "configs") -> Dict[str, Any]:
    """Load all configuration files"""
    configs = {}
    
    # Load environment config
    env_path = os.path.join(config_dir, "env.yaml")
    if os.path.exists(env_path):
        configs['env'] = load_config(env_path)
    
    # Load model config  
    model_path = os.path.join(config_dir, "model.yaml")
    if os.path.exists(model_path):
        configs['model'] = load_config(model_path)
    
    # Load training config
    train_path = os.path.join(config_dir, "train.yaml")
    if os.path.exists(train_path):
        configs['train'] = load_config(train_path)
    
    return configs


def get_env_params(config_dir: str = "configs") -> Dict[str, Any]:
    """Get environment parameters from config"""
    configs = load_all_configs(config_dir)
    
    # Default values (fallback)
    defaults = {
        'num_agents': 4,
        'agent_radius': 0.5,
        'corridor_width': 20.0,
        'corridor_height': 10.0,
        'bottleneck_width': 1.2,
        'bottleneck_position': 10.0,
        'sensing_radius': 3.0,
        'max_timesteps': 300
    }
    
    # Override with config values if available
    if 'env' in configs and 'environment' in configs['env']:
        env_config = configs['env']['environment']
        defaults.update(env_config)
    
    return defaults