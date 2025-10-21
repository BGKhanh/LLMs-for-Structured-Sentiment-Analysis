# src/config/loader.py

"""
Configuration loading and merging utilities.

Handles loading YAML configs, merging base with experiment configs,
and resolving file paths.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Any
from .schema import Config


def load_yaml(path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        path: Path to YAML file
        
    Returns:
        Dictionary from YAML
        
    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML is invalid
    """
    yaml_path = Path(path)
    
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {path}: {e}")



def resolve_paths(config_dict: Dict[str, Any], project_root: Path) -> Dict[str, Any]:
    """
    Resolve relative paths to absolute paths.
    
    Args:
        config_dict: Configuration dictionary
        project_root: Project root directory
        
    Returns:
        Config dict with resolved paths
    """
    # Resolve dataset_path
    if 'data' in config_dict and 'dataset_path' in config_dict['data']:
        dataset_path = Path(config_dict['data']['dataset_path'])
        if not dataset_path.is_absolute():
            config_dict['data']['dataset_path'] = str(project_root / dataset_path)
    
    # Resolve examples_pool_path
    if 'prompt' in config_dict and config_dict['prompt'].get('examples_pool_path'):
        pool_path = Path(config_dict['prompt']['examples_pool_path'])
        if not pool_path.is_absolute():
            config_dict['prompt']['examples_pool_path'] = str(project_root / pool_path)
    
    # Resolve output_dir
    if 'output' in config_dict and 'output_dir' in config_dict['output']:
        output_dir = Path(config_dict['output']['output_dir'])
        if not output_dir.is_absolute():
            config_dict['output']['output_dir'] = str(project_root / output_dir)
    
    return config_dict


def load_config(
    config_path: str,
    base_config_path: str = "configs/base.yaml",
    project_root: Optional[str] = None
) -> Config:
    """
    Load configuration with base config merging.
    
    Args:
        config_path: Path to experiment config file
        base_config_path: Path to base config (default: configs/base.yaml)
        project_root: Project root directory (default: auto-detected)
        
    Returns:
        Loaded and merged Config object
        
    Example:
        >>> config = load_config("configs/experiments/exp_001.yaml")
        >>> print(config.model.model_id)
    """
    # Auto-detect project root if not provided
    if project_root is None:
        config_file = Path(config_path).resolve()
        # Assume project root is 2 levels up from configs/experiments/
        project_root = config_file.parent.parent.parent
    else:
        project_root = Path(project_root)
    
    print(f"Loading config from: {config_path}")
    
    # Load experiment config
    exp_dict = load_yaml(config_path)
    
    # Resolve relative paths
    exp_resolved_dict = resolve_paths(exp_dict, project_root)
    
    # Convert to Config object
    config = Config.from_dict(exp_resolved_dict)
    
    print(f"✅ Config loaded: {config.experiment.name}")
    
    return config