# src/config/schema.py

"""
Configuration schema definitions using dataclasses.

Defines the complete structure of config.yaml with type hints and defaults.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Literal, Dict, Any, List

class FlexibleConfig:
    """
    Base class for configs that allow extra fields (dynamic kwargs).
    Unknown arguments passed to __init__ are stored in self.extra_args.
    """
    def __init__(self, **kwargs):
        # Store definition-based fields
        names = set([f.name for f in self.__dataclass_fields__.values()])
        
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)
            else:
                # Store unknown fields dynamically
                # We simply set them as attributes so they act like normal fields
                setattr(self, k, v)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert all attributes (static + dynamic) to dictionary."""
        # Get static fields
        data = asdict(self) # type: ignore
        
        # Merge with dynamic attributes that are not in dataclass fields
        # Note: asdict already handles dataclass fields. 
        # We need to add attributes that were added dynamically in __init__
        # but NOT duplicate what asdict already does.
        
        # Simpler approach for this specific use case:
        # Just return the instance's __dict__, but filtered for internal Python stuff
        result = {}
        for k, v in self.__dict__.items():
            if not k.startswith('__'):
                result[k] = v
        return result
    
@dataclass
class ExperimentConfig:
    """Experiment metadata configuration."""
    name: str = "unnamed_experiment"
    description: str = ""
    version: str = "1.0"


@dataclass
class ModelInitConfig(FlexibleConfig):
    """
    Parameters for Model Initialization (load_model / from_pretrained).
    Allows flexible extra arguments (e.g., attn_implementation, quantization_config).
    """
    # === STANDARD PARAMS (Validated) ===
    model_id: str = "google/gemma-3-4b-it"
    dtype: Literal["float32", "float16", "bfloat16", "auto"] = "float32"
    device_map: str = "auto"
    trust_remote_code: bool = True
    
    # === OPTIONAL STANDARD PARAMS ===
    cache_dir: Optional[str] = None
    token: Optional[str] = None
    revision: str = "main"
    
    # Extra params are handled by FlexibleConfig.__init__.

@dataclass
class ModelGenerationConfig(FlexibleConfig):
    """
    Parameters for Model Generation (model.generate).
    Allows flexible extra arguments (e.g., min_p, repetition_penalty, enable_thinking).
    """
    # === STANDARD PARAMS (Validated) ===
    max_new_tokens: int = 4096
    do_sample: bool = False
    
    # === COMMON SAMPLING PARAMS ===
    temperature: float = 0.1
    top_p: float = 0.95
    top_k: int = 50
    
    # === SPECIAL PARAMS ===
    enable_thinking: bool = False  # Moved here as it affects generation flow

    # Extra params (min_p, guidance_scale, etc.) are handled by FlexibleConfig.__init__


@dataclass
class ModelConfig:
    """
    Model configuration.
    
    Design: Common parameters as explicit fields,
    rare parameters via additional_model_kwargs.
    """
    # === REQUIRED ===
    name: Literal["gemma", "mistral", "qwen", "llama", "seallm", "vistral", "llama4", "llama3", "vinallama"] = "gemma"
    # Sub-configs
    init_args: ModelInitConfig = field(default_factory=ModelInitConfig)
    generation_args: ModelGenerationConfig = field(default_factory=ModelGenerationConfig)

    def __post_init__(self):
        # Ensure sub-configs are converted to objects if they are dicts 
        # (This happens when loading from YAML via simple dict unpacking)
        if isinstance(self.init_args, dict):
            self.init_args = ModelInitConfig(**self.init_args)
        if isinstance(self.generation_args, dict):
            self.generation_args = ModelGenerationConfig(**self.generation_args)


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
    num_workers: Optional[int] = 2 

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
                      "zero_shot_cot", "plan_and_solve"] = "rereading"
    language: Literal["vi", "en"] = "vi"
    
    # Few-shot specific parameters
    n_shot: int = 0
    examples_pool_path: Optional[str] = None
    
    # Plan-and-solve specific parameters
    plus_mode: bool = False  # True for PS+, False for PS
    add_method: Literal["none", "CoT", "PaS"] = "none" # none for vanilla RE2, CoT for RE2+CoT, PaS for RE2+PaS

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
    random_seed: int = 42
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    cleanup_frequency: int = 0  # GPU cleanup every N batches
    
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
            random_seed=config_dict.get('random_seed', 42),
            experiment=ExperimentConfig(**config_dict.get('experiment', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            data=DataConfig(**config_dict.get('data', {})),
            prompt=PromptConfig(**config_dict.get('prompt', {})),
            output=OutputConfig(**config_dict.get('output', {})),
            cleanup_frequency=config_dict.get('cleanup_frequency', 0)
        )
    
    def to_dict(self) -> dict:
        """Convert Config to dictionary."""
        # Custom to_dict to handle FlexibleConfig properly
        d = asdict(self)
        # Re-inject dynamic fields from model config
        d['model']['init_args'] = self.model.init_args.to_dict()
        d['model']['generation_args'] = self.model.generation_args.to_dict()
        return d