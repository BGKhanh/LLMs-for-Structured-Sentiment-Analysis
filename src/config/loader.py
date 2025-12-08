# src/config/loader.py

"""
Configuration loading and merging utilities.

Handles loading YAML configs, merging base with experiment configs,
and resolving file paths.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Any, Union
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
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
        
    with open(path, 'r', encoding='utf-8') as f:
        try:
            config_dict = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")
            
    return config_dict



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
    if 'data' in config_dict:
        data_conf = config_dict['data']
        # List of path fields in DataConfig
        path_fields = ['train_dataset_path', 'dev_dataset_path', 'test_dataset_path']
        
        for field in path_fields:
            if field in data_conf and data_conf[field]:
                path = Path(data_conf[field])
                if not path.is_absolute():
                    data_conf[field] = str(project_root / path)
    
    # 2. Resolve Prompt paths
    if 'prompt' in config_dict:
        prompt_conf = config_dict['prompt']
        if 'examples_pool_path' in prompt_conf and prompt_conf['examples_pool_path']:
            path = Path(prompt_conf['examples_pool_path'])
            if not path.is_absolute():
                prompt_conf['examples_pool_path'] = str(project_root / path)
    
    # 3. Resolve Output paths
    if 'output' in config_dict:
        output_conf = config_dict['output']
        if 'output_dir' in output_conf and output_conf['output_dir']:
            path = Path(output_conf['output_dir'])
            if not path.is_absolute():
                output_conf['output_dir'] = str(project_root / path)
    
    return config_dict


def load_config(
    config_path: Union[str, Path],
    project_root: Optional[Union[str, Path]] = None
) -> Config:
    """
    Load Config object from YAML file.
    
    Args:
        config_path: Path to experiment config file
        project_root: Project root directory (default: auto-detected)
        
    Returns:
        Validated Config object
    """
    config_file = Path(config_path).resolve()
    
    # Auto-detect project root if not provided
    # Assumption: configs are usually in <root>/configs/...
    if project_root is None:
        # Try to find 'src' directory walking up
        current = config_file.parent
        for _ in range(3): # Look up 3 levels
            if (current / "src").exists():
                project_root = current
                break
            current = current.parent
        
        # Fallback if not found
        if project_root is None:
            project_root = Path(".")
    else:
        project_root = Path(project_root)
    
    print(f"📂 Loading config from: {config_file}")
    print(f"📍 Project root detected: {project_root}")
    
    # 1. Load raw dict
    config_dict = load_yaml(config_file)
    
    # 2. Resolve paths
    config_dict = resolve_paths(config_dict, project_root)
    
    # 3. Convert to Config object
    # Config.from_dict handles nested ModelConfig creation (init_args/generation_args)
    try:
        config = Config.from_dict(config_dict)
    except Exception as e:
        raise ValueError(f"Error creating Config object: {e}")
        
    print(f"✅ Config loaded: {config.experiment.name}")
    return config