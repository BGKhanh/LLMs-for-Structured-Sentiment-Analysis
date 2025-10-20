# src/config/schema.py

"""
Configuration schema definitions using dataclasses.

Defines the complete structure of config.yaml with type hints and defaults.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, Any


@dataclass
class ExperimentConfig:
    """Experiment metadata configuration."""
    name: str = "unnamed_experiment"
    description: str = ""
    version: str = "1.0"



@dataclass
class ModelConfig:
    """
    Model configuration.
    
    Design: Common parameters as explicit fields,
    rare parameters via additional_model_kwargs.
    """
    # === REQUIRED ===
    name: Literal["gemma", "mistral", "qwen", "llama"] = "gemma"
    model_id: str = "google/gemma-3-4b-it"
    
    # === COMMON MODEL LOADING PARAMS ===
    dtype: Literal["float32", "float16", "bfloat16", "auto"] = "float32"
    device_map: str = "auto"
    trust_remote_code: bool = True
    
    # === COMMON GENERATION PARAMS ===
    max_new_tokens: int = 4096
    do_sample: bool = False
    temperature: float = 0.1
    top_p: float = 0.95
    top_k: int = 50
    
    # === OPTIONAL BUT USEFUL ===
    cache_dir: Optional[str] = None
    token: Optional[str] = None  # HF token
    revision: str = "main"
    
    # === ADVANCED/RARE PARAMS (as dict) ===
    additional_model_kwargs: Dict[str, Any] = field(default_factory=dict)
    # Can include: quantization_config, attn_implementation, 
    # max_memory, offload_folder, use_safetensors, etc.
    
    def get_from_pretrained_kwargs(self) -> Dict[str, Any]:
        """
        Build kwargs for from_pretrained() call.
        
        Returns:
            Dictionary with all model loading parameters
        """
        import torch
        
        # Start with common parameters
        kwargs = {
            "device_map": self.device_map,
            "trust_remote_code": self.trust_remote_code,
        }
        
        # Handle dtype
        if self.dtype == "auto":
            kwargs["dtype"] = "auto"
        else:
            kwargs["dtype"] = getattr(torch, self.dtype)
        
        # Add optional parameters if specified
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir
        if self.token:
            kwargs["token"] = self.token
        if self.revision != "main":
            kwargs["revision"] = self.revision
        
        # Merge additional kwargs (user can override or add more params)
        kwargs.update(self.additional_model_kwargs)
        
        return kwargs
    
    def get_generation_kwargs(self) -> Dict[str, Any]:
        """
        Build kwargs for generate() call.
        
        Returns:
            Dictionary with generation parameters
        """
        kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
        }
        
        # Add sampling parameters if do_sample is True
        if self.do_sample:
            kwargs["temperature"] = self.temperature
            kwargs["top_p"] = self.top_p
            kwargs["top_k"] = self.top_k
        
        return kwargs

@dataclass
class DataConfig:
    """Data loading configuration."""
    # Dataset selection
    dataset: Literal["train", "dev", "test"] = "dev"
    
    train_dataset_path: str = "data/train.json"
    dev_dataset_path: str = "data/dev.json"
    test_dataset_path: str = "data/test.json"
    
    # Examples pool for few-shot (references dataset name)
    examples_pool: Literal["train", "dev", "test"] = "train"
    
    batch_size: int = 8
    num_samples: Optional[int] = None  # None means all samples

    def get_dataset_path(self) -> str:
        """Get the selected dataset path."""
        if self.dataset == "train":
            return self.train_dataset_path
        elif self.dataset == "dev":
            return self.dev_dataset_path
        else:  # test
            return self.test_dataset_path
    
    def get_examples_pool_path(self) -> str:
        """Get the examples pool path."""
        if self.examples_pool == "train":
            return self.train_dataset_path
        elif self.examples_pool == "dev":
            return self.dev_dataset_path
        else:  # test
            return self.test_dataset_path


@dataclass
class PromptConfig:
    """Prompt technique configuration."""
    technique: Literal["rereading", "few_shot", "few_shot_cot", 
                      "zero_shot_cot", "plan_solve"] = "rereading"
    language: Literal["vi", "en"] = "vi"
    
    # Few-shot specific parameters
    n_shot: int = 0
    examples_pool_path: Optional[str] = None
    
    # Plan-and-solve specific parameters
    plus_mode: bool = False  # True for PS+, False for PS


@dataclass
class OutputConfig:
    """Output configuration."""
    output_dir: str = "results/"
    output_file: str = "results.json"
    save_metadata: bool = True
    save_config_copy: bool = True
    overwrite: bool = False


@dataclass
class Config:
    """Complete configuration for inference."""
    seed: int = 42
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    cleanup_frequency: int = 10  # GPU cleanup every N batches
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Config':
        """
        Create Config from nested dictionary.
        
        Args:
            config_dict: Dictionary from YAML
            
        Returns:
            Config object
        """
        return cls(
            seed=config_dict.get('seed', 42),
            experiment=ExperimentConfig(**config_dict.get('experiment', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            data=DataConfig(**config_dict.get('data', {})),
            prompt=PromptConfig(**config_dict.get('prompt', {})),
            output=OutputConfig(**config_dict.get('output', {})),
            cleanup_frequency=config_dict.get('cleanup_frequency', 10)
        )
    
    def to_dict(self) -> dict:
        """Convert Config to dictionary."""
        from dataclasses import asdict
        return asdict(self)