# src/config/training_loader.py

"""
Training configuration loading utilities.

Similar to loader.py but for training configs.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from .training_schema import TrainingConfig


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
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
        
    with open(path, 'r', encoding='utf-8') as f:
        try:
            config_dict = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")
            
    return config_dict


def resolve_training_paths(config_dict: Dict[str, Any], project_root: Path) -> Dict[str, Any]:
    """
    Resolve relative paths to absolute paths.
    
    Args:
        config_dict: Configuration dictionary
        project_root: Project root directory
        
    Returns:
        Config dict with resolved paths
    """
    # 1. Resolve model path (if it's local path, not HF hub)
    if 'model' in config_dict:
        model_conf = config_dict['model']
        if 'model_name_or_path' in model_conf:
            model_path = Path(model_conf['model_name_or_path'])
            # Only resolve if it's a relative path (not HF hub ID like "google/gemma")
            if not "/" in model_conf['model_name_or_path'] or model_path.exists():
                if not model_path.is_absolute() and model_path.exists():
                    model_conf['model_name_or_path'] = str(project_root / model_path)
    
    # 2. Resolve data paths
    if 'data' in config_dict:
        data_conf = config_dict['data']
        path_fields = ['train_dataset_path', 'eval_dataset_path']
        
        for field in path_fields:
            if field in data_conf and data_conf[field]:
                path = Path(data_conf[field])
                if not path.is_absolute():
                    data_conf[field] = str(project_root / path)
    
    # 3. Resolve prompt examples pool path (reused from inference schema)
    if 'prompt' in config_dict:
        prompt_conf = config_dict['prompt']
        if 'examples_pool_path' in prompt_conf and prompt_conf['examples_pool_path']:
            path = Path(prompt_conf['examples_pool_path'])
            if not path.is_absolute():
                prompt_conf['examples_pool_path'] = str(project_root / path)
    
    # 4. Resolve training output dir
    if 'training' in config_dict:
        training_conf = config_dict['training']
        if 'output_dir' in training_conf and training_conf['output_dir']:
            path = Path(training_conf['output_dir'])
            if not path.is_absolute():
                training_conf['output_dir'] = str(project_root / path)
    
    return config_dict


def load_training_config(
    config_path: Union[str, Path],
    project_root: Optional[Union[str, Path]] = None
) -> TrainingConfig:
    """
    Load TrainingConfig object from YAML file.
    
    Args:
        config_path: Path to training config file
        project_root: Project root directory (default: auto-detected)
        
    Returns:
        Validated TrainingConfig object
    """
    config_file = Path(config_path).resolve()
    
    # Auto-detect project root
    if project_root is None:
        current = config_file.parent
        for _ in range(3):
            if (current / "src").exists():
                project_root = current
                break
            current = current.parent
        
        if project_root is None:
            project_root = Path(".")
    else:
        project_root = Path(project_root)
    
    print(f"📂 Loading training config from: {config_file}")
    print(f"📍 Project root detected: {project_root}")
    
    # 1. Load raw dict
    config_dict = load_yaml(config_file)
    
    # 2. Resolve paths
    config_dict = resolve_training_paths(config_dict, project_root)
    
    # 3. Convert to TrainingConfig object
    try:
        config = TrainingConfig.from_dict(config_dict)
    except Exception as e:
        raise ValueError(f"Error creating TrainingConfig object: {e}")
        
    print(f"✅ Training config loaded: {config.experiment_name}")
    return config