# src/utils/save_result.py

"""
Results saving utilities.

Saves inference results with organized directory structure:
results/{model_name}/{experiment_name}/
    ├── result_rank0.json      # Processed results (Rank 0)
    ├── result_rank1.json      # Processed results (Rank 1)
    ├── config.json            # Full config copy (Shared)
    ├── metadata_rank0.json    # Runtime info only
    └── debug_info_rank0.json  # Prompts + raw responses
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import sys
import re


def save_experiment_results(
    results: List[Dict[str, Any]],
    config: 'Config',
    statistics: Optional[Dict[str, Any]] = None,
    raw_results: Optional[List[Dict[str, Any]]] = None,
    results_dir: str = "results",
    rank: Optional[int] = None
) -> Dict[str, str]:
    """
    Save experiment results with organized structure.
    
    Directory structure:
        results/
        └── {model_name}/
            └── {experiment_name}/
                ├── result.json      # Inference results
                ├── config.json      # Config copy
                ├── metadata.json    # Runtime metadata (simplified)
                └── debug_info.json  # Prompts + raw responses
    
    Args:
        results: List of inference results (SemEval format)
        config: Config object
        statistics: Optional statistics dict from pipeline
        raw_results: Optional raw results from pipeline (for debug_info)
        results_dir: Base results directory
        rank: Process rank (for multi-GPU). If None, assumes single process.
    Returns:
        Dict with paths to saved files:
        {
            'result_file': '...',
            'metadata_file': '...',
            'config_file': '...',
            'debug_file': '...',
            'experiment_dir': '...'
        }
    
    Raises:
        OSError: If cannot create directories or write files
    """
    # Build directory structure
    model_name = config.model.name
    experiment_name = config.experiment.name
    
    experiment_dir = Path(results_dir) / model_name / experiment_name
    
    if rank is None or rank == 0:
        print(f"\n📁 Creating experiment directory: {experiment_dir}")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    if rank is None or rank == 0:
        print(f"✅ Directory ready: {experiment_dir}")
        
    suffix = f"_rank{rank}" if rank is not None else ""
    # Define file paths
    result_file = experiment_dir / f"result{suffix}.json"
    metadata_file = experiment_dir / f"metadata{suffix}.json"
    debug_file = experiment_dir / f"debug_info{suffix}.json"
    
    config_file = experiment_dir / "config.json" # Shared config file
    # Check overwrite
    if not config.output.overwrite:
        if result_file.exists():
            raise FileExistsError(
                f"Result file already exists: {result_file}\n"
                f"Set output.overwrite=true in config to overwrite"
            )
    
    # Save results
    if rank is None or rank == 0: print(f"\n💾 Saving results (Rank {rank})...")
    _save_json(result_file, results, f"Results (Rank {rank})")
    
    # Save metadata
    if config.output.save_metadata:
        if rank is None or rank == 0: print(f"\n💾 Saving metadata...")
        metadata = _build_metadata(config, results, statistics, rank)
        _save_json(metadata_file, metadata, f"Metadata (Rank {rank})")
    
    # Save debug info 
    if rank is None or rank == 0: print(f"\n💾 Saving debug info (Rank {rank})...")
    debug_path = save_debug_info(
        raw_results=raw_results or [],
        experiment_dir=experiment_dir,
        technique=config.prompt.technique,
        suffix=suffix # Pass suffix to helper
    )
    
    config_saved_path = None
    if config.output.save_config_copy:
        # Only Main Process (Rank 0 or Single GPU) saves config
        if rank is None or rank == 0:
            print(f"\n💾 Saving config copy...")
            config_dict = config.to_dict()
            _save_json(config_file, config_dict, "Config")
            config_saved_path = str(config_file)
            
    # Return saved paths
    saved_paths = {
        'result_file': str(result_file),
        'metadata_file': str(metadata_file) if config.output.save_metadata else None,
        'config_file': str(config_file) if config.output.save_config_copy else None,
        'debug_file': debug_path,
        'experiment_dir': str(experiment_dir)
    }
    
    if rank is None or rank == 0:
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
    statistics: Optional[Dict[str, Any]],
    rank: Optional[int] = None
) -> Dict[str, Any]:
    """
    Build metadata dictionary (SIMPLIFIED - runtime info only).
    
    Rationale: Config info is redundant since config.json is saved.
    Only save runtime-specific information here.
    
    Args:
        config: Config object
        results: Inference results
        statistics: Optional statistics from pipeline
        
    Returns:
        Metadata dictionary with runtime info only
    """
    import torch
    import transformers
    
    # Calculate statistics if not provided
    if statistics is None:
        statistics = _calculate_statistics(results)
    
    # Build metadata
    metadata = {
        # Timestamp (unique to this run)
        "timestamp": datetime.now().isoformat(),
        "rank": rank if rank is not None else 0, # Log rank
        # Statistics (runtime metrics)
        "statistics": statistics,
        
        # Environment info (runtime environment)
        "environment": {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device": torch.cuda.get_device_name(rank) if rank is not None and torch.cuda.is_available() and rank < torch.cuda.device_count() else "unknown"
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

def save_debug_info(
    raw_results: List[Dict[str, Any]],
    experiment_dir: Path,
    technique: str,
    suffix: str = ""
) -> str:
    """
    Save debug information (prompts + raw responses).
    
    Handles both single-stage and multi-stage techniques.
    Always saves - this is mandatory for debugging and analysis.
    
    Args:
        raw_results: List from pipeline.raw_results
        experiment_dir: Experiment directory path
        technique: Prompt technique name
        
    Returns:
        Path to saved debug_info.json file
        
    Structure for single-stage:
        {
            "technique": "rereading",
            "is_multi_stage": false,
            "shared_prompts": {
                "system_prompt": "..."  # Same for all samples
            },
            "samples": [
                {
                    "sent_id": "...",
                    "user_prompt": "...",
                    "raw_response": "...",
                    "generation_time": 0.5
                }
            ]
        }
    
    Structure for multi-stage (Zero-shot CoT):
        {
            "technique": "zero_shot_cot",
            "is_multi_stage": true,
            "stage_prompts": {
                "stage_1": {
                    "system_prompt": "...",
                    "purpose": "Generate reasoning"
                },
                "stage_2": {
                    "system_prompt": "...",
                    "purpose": "Extract structured output"
                }
            },
            "samples": [
                {
                    "sent_id": "...",
                    "stage_1": {
                        "user_prompt": "...",
                        "raw_response": "...",
                        "generation_time": 0.3
                    },
                    "stage_2": {
                        "user_prompt": "...",
                        "raw_response": "...",
                        "generation_time": 0.4
                    }
                }
            ]
        }
    """
    
    filename = f"debug_info{suffix}.json" # Use dynamic filename
    debug_file = experiment_dir / filename
    if not raw_results:
        print("   ⚠️  No raw results to save in debug_info")
        # Create empty debug file
        debug_data = {
            "technique": technique,
            "is_multi_stage": False,
            "samples": []
        }
        _save_json(debug_file, debug_data, "Debug info (empty)")
        return str(debug_file)
    
    # Detect if multi-stage
    is_multi_stage = technique == "zero_shot_cot"
    
    debug_data = {
        "technique": technique,
        "is_multi_stage": is_multi_stage,
        "samples": []
    }
    
    if is_multi_stage:
        # === MULTI-STAGE (Zero-shot CoT) ===
        
        # Extract stage-specific shared prompts from first successful sample
        first_success = next((r for r in raw_results if r.get('success')), None)
        
        if first_success and 'stage_1' in first_success and 'stage_2' in first_success:
            debug_data["stage_prompts"] = {
                "stage_1": {
                    "system_prompt": first_success['stage_1'].get('system_prompt', ''),
                    "purpose": "Generate reasoning"
                },
                "stage_2": {
                    "system_prompt": first_success['stage_2'].get('system_prompt', ''),
                    "purpose": "Extract structured output"
                }
            }
        
        # Per-sample data (multi-stage)
        for result in raw_results:
            if not result.get('success'):
                continue
            
            sample_data = {
                "sent_id": result.get('sent_id', 'unknown')
            }
            
            # Stage 1 data
            if 'stage_1' in result:
                sample_data["stage_1"] = {
                    "user_prompt": result['stage_1'].get('user_prompt', ''),
                    "raw_response": result['stage_1'].get('raw_response', ''),
                    "generation_time": result['stage_1'].get('generation_time', 0.0),
                    "input_tokens": result['stage_1'].get('input_tokens', 0),      
                    "output_tokens": result['stage_1'].get('output_tokens', 0)     
                }
            
            # Stage 2 data
            if 'stage_2' in result:
                sample_data["stage_2"] = {
                    "user_prompt": result['stage_2'].get('user_prompt', ''),
                    "raw_response": result['stage_2'].get('raw_response', ''),
                    "generation_time": result['stage_2'].get('generation_time', 0.0),
                    "input_tokens": result['stage_2'].get('input_tokens', 0),      
                    "output_tokens": result['stage_2'].get('output_tokens', 0)     
                }
            
            debug_data["samples"].append(sample_data)
    
    else:
        # === SINGLE-STAGE ===
        
        # Extract shared system prompt from first successful sample
        first_success = next((r for r in raw_results if r.get('success')), None)
        
        if first_success and 'system_prompt' in first_success:
            debug_data["shared_prompts"] = {
                "system_prompt": first_success['system_prompt']
            }
        
        # Per-sample data (single-stage)
        for result in raw_results:
            if not result.get('success'):
                continue
            
            debug_data["samples"].append({
                "sent_id": result.get('sent_id', 'unknown'),
                "user_prompt": result.get('user_prompt', ''),
                "raw_response": result.get('raw_response', ''),
                "generation_time": result.get('generation_time', 0.0),
                "input_tokens": result.get('input_tokens', 0),      
                "output_tokens": result.get('output_tokens', 0)     
            })
    
    # Save debug info
    _save_json(debug_file, debug_data, f"Debug info {suffix}")
    
    print(f"   📊 Saved {len(debug_data['samples'])} samples to debug_info.json")
    
    return str(debug_file)

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
            result_files = list(exp_dir.glob("result*.json"))
            if result_files:
                # Lấy file đầu tiên tìm thấy làm đại diện
                main_result_file = result_files[0]
                experiments.append({
                    'model_name': model_dir.name,
                    'experiment_name': exp_dir.name,
                    'path': str(exp_dir),
                    'result_file': str(main_result_file),
                    'has_metadata': list(exp_dir.glob("metadata*.json")) != [],
                    'has_config': (exp_dir / "config.json").exists(),
                    'has_debug_info': list(exp_dir.glob("debug_info*.json")) != [],  
                })
    
    return experiments


def load_experiment_results(
    model_name: str,
    experiment_name: str,
    results_dir: str = "results"
) -> Dict[str, Any]:
    """
    Load saved experiment results (Fix: Supports multi-gpu filenames).
    """
    exp_dir = Path(results_dir) / model_name / experiment_name
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment not found: {exp_dir}")
    
    data = {}
    
    # === FIX 4: Load Priority: result.json -> result_rank0.json ===
    if (exp_dir / "result.json").exists():
        result_file = exp_dir / "result.json"
    elif (exp_dir / "result_rank0.json").exists():
        result_file = exp_dir / "result_rank0.json"
        print(f"ℹ️ Loading distributed results from Rank 0: {result_file.name}")
    else:
        raise FileNotFoundError(f"No result file found in {exp_dir}")
    
    with open(result_file, 'r', encoding='utf-8') as f:
        data['results'] = json.load(f)
    
    # Load metadata (Tương tự)
    if (exp_dir / "metadata.json").exists():
        meta_path = exp_dir / "metadata.json"
    elif (exp_dir / "metadata_rank0.json").exists():
        meta_path = exp_dir / "metadata_rank0.json"
    else:
        meta_path = None
        
    if meta_path:
        with open(meta_path, 'r', encoding='utf-8') as f:
            data['metadata'] = json.load(f)
    
    # Load config
    config_file = exp_dir / "config.json"
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            data['config'] = json.load(f)
            
    return data