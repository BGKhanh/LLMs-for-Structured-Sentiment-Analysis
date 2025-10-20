# src/utils/save_result.py

"""
Results saving utilities.

Saves inference results with organized directory structure:
results/{model_name}/{experiment_name}/
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import sys


def save_experiment_results(
    results: List[Dict[str, Any]],
    config: 'Config',
    statistics: Optional[Dict[str, Any]] = None,
    results_dir: str = "results"
) -> Dict[str, str]:
    """
    Save experiment results with organized structure.
    
    Directory structure:
        results/
        └── {model_name}/
            └── {experiment_name}/
                ├── result.json      # Inference results
                ├── metadata.json    # Experiment metadata
                └── config.json      # Config copy
    
    Args:
        results: List of inference results (SemEval format)
        config: Config object
        statistics: Optional statistics dict
        results_dir: Base results directory
        
    Returns:
        Dict with paths to saved files:
        {
            'result_file': '...',
            'metadata_file': '...',
            'config_file': '...',
            'experiment_dir': '...'
        }
    
    Raises:
        OSError: If cannot create directories or write files
    """
    # Build directory structure
    model_name = config.model.name
    experiment_name = config.experiment.name
    
    experiment_dir = Path(results_dir) / model_name / experiment_name
    
    # Create directory
    print(f"\n📁 Creating experiment directory: {experiment_dir}")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Directory ready: {experiment_dir}")
    
    # Define file paths
    result_file = experiment_dir / "result.json"
    metadata_file = experiment_dir / "metadata.json"
    config_file = experiment_dir / "config.json"
    
    # Check overwrite
    if not config.output.overwrite:
        if result_file.exists():
            raise FileExistsError(
                f"Result file already exists: {result_file}\n"
                f"Set output.overwrite=true in config to overwrite"
            )
    
    # Save results
    print(f"\n💾 Saving results...")
    _save_json(result_file, results, "Results")
    
    # Save metadata
    if config.output.save_metadata:
        print(f"\n💾 Saving metadata...")
        metadata = _build_metadata(config, results, statistics)
        _save_json(metadata_file, metadata, "Metadata")
    
    # Save config copy
    if config.output.save_config_copy:
        print(f"\n💾 Saving config copy...")
        config_dict = config.to_dict()
        _save_json(config_file, config_dict, "Config")
    
    # Return saved paths
    saved_paths = {
        'result_file': str(result_file),
        'metadata_file': str(metadata_file) if config.output.save_metadata else None,
        'config_file': str(config_file) if config.output.save_config_copy else None,
        'experiment_dir': str(experiment_dir)
    }
    
    print(f"\n✅ All files saved to: {experiment_dir}")
    
    return saved_paths


def _save_json(file_path: Path, data: Any, label: str) -> None:
    """
    Save data to JSON file with pretty printing.
    
    Args:
        file_path: Path to save file
        data: Data to save
        label: Label for logging
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"   ✅ {label} saved: {file_path.name}")
    except Exception as e:
        raise OSError(f"Failed to save {label} to {file_path}: {e}")


def _build_metadata(
    config: 'Config',
    results: List[Dict[str, Any]],
    statistics: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Build metadata dictionary.
    
    Args:
        config: Config object
        results: Inference results
        statistics: Optional statistics from pipeline
        
    Returns:
        Metadata dictionary
    """
    import torch
    import transformers
    
    # Calculate statistics if not provided
    if statistics is None:
        statistics = _calculate_statistics(results)
    
    # Build metadata
    metadata = {
        # Experiment info
        "experiment": {
            "name": config.experiment.name,
            "description": config.experiment.description,
            "version": config.experiment.version,
            "timestamp": datetime.now().isoformat(),
        },
        
        # Model info
        "model": {
            "name": config.model.name,
            "model_id": config.model.model_id,
            "dtype": config.model.dtype,
            "device_map": config.model.device_map,
            "max_tokens": config.model.max_tokens,
        },
        
        # Data info
        "data": {
            "dataset": config.data.dataset,
            "dataset_path": config.data.get_dataset_path(),
            "batch_size": config.data.batch_size,
            "n_sample": config.data.n_sample,
        },
        
        # Prompt info
        "prompt": {
            "technique": config.prompt.technique,
            "language": config.prompt.language,
            "n_shot": config.prompt.n_shot,
        },
        
        # Statistics
        "statistics": statistics,
        
        # Environment info
        "environment": {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() and torch.cuda.device_count() > 0 else None,
        }
    }
    
    return metadata


def _calculate_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate statistics from results.
    
    Args:
        results: List of inference results
        
    Returns:
        Statistics dictionary
    """
    total_samples = len(results)
    
    # Count successful vs failed
    successful = sum(1 for r in results if 'error' not in r)
    failed = total_samples - successful
    
    # Count opinions
    total_opinions = sum(len(r.get('opinions', [])) for r in results)
    avg_opinions = total_opinions / total_samples if total_samples > 0 else 0
    
    return {
        "total_samples": total_samples,
        "successful": successful,
        "failed": failed,
        "success_rate": successful / total_samples if total_samples > 0 else 0,
        "total_opinions": total_opinions,
        "avg_opinions_per_sample": round(avg_opinions, 2)
    }


def list_experiments(model_name: Optional[str] = None, results_dir: str = "results") -> List[Dict[str, str]]:
    """
    List all saved experiments.
    
    Args:
        model_name: Filter by model name (None = all models)
        results_dir: Base results directory
        
    Returns:
        List of experiment info dicts
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        return []
    
    experiments = []
    
    # Iterate through model directories
    for model_dir in results_path.iterdir():
        if not model_dir.is_dir():
            continue
        
        if model_name and model_dir.name != model_name:
            continue
        
        # Iterate through experiment directories
        for exp_dir in model_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            # Check if has result.json
            result_file = exp_dir / "result.json"
            if result_file.exists():
                experiments.append({
                    'model_name': model_dir.name,
                    'experiment_name': exp_dir.name,
                    'path': str(exp_dir),
                    'result_file': str(result_file),
                    'has_metadata': (exp_dir / "metadata.json").exists(),
                    'has_config': (exp_dir / "config.json").exists(),
                })
    
    return experiments


def load_experiment_results(
    model_name: str,
    experiment_name: str,
    results_dir: str = "results"
) -> Dict[str, Any]:
    """
    Load saved experiment results.
    
    Args:
        model_name: Model name (e.g., "gemma")
        experiment_name: Experiment name (e.g., "dev_rereading")
        results_dir: Base results directory
        
    Returns:
        Dict with:
        {
            'results': [...],
            'metadata': {...},
            'config': {...}
        }
    
    Raises:
        FileNotFoundError: If experiment not found
    """
    exp_dir = Path(results_dir) / model_name / experiment_name
    
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment not found: {exp_dir}")
    
    # Load files
    data = {}
    
    # Load results (required)
    result_file = exp_dir / "result.json"
    if not result_file.exists():
        raise FileNotFoundError(f"Result file not found: {result_file}")
    
    with open(result_file, 'r', encoding='utf-8') as f:
        data['results'] = json.load(f)
    
    # Load metadata (optional)
    metadata_file = exp_dir / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r', encoding='utf-8') as f:
            data['metadata'] = json.load(f)
    else:
        data['metadata'] = None
    
    # Load config (optional)
    config_file = exp_dir / "config.json"
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            data['config'] = json.load(f)
    else:
        data['config'] = None
    
    return data