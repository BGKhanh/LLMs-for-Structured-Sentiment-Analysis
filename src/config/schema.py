# src/config/schema.py

"""
Configuration schema definitions using dataclasses.

Defines the complete structure of config.yaml with type hints and defaults.
"""

from dataclasses import dataclass, field, asdict, MISSING
from typing import Optional, Literal, Dict, Any, List

class FlexibleConfig:
    """
    Base class for configs that allow extra fields (dynamic kwargs).

    Behaviour:
    - When instantiated, apply dataclass defaults (if any).
    - Override defaults with kwargs passed in.
    - Unknown kwargs are stored both as attributes and in self._extra_args.
    - to_dict() returns a merged dictionary of dataclass fields + extra fields.
    """

    def __init__(self, **kwargs):
        # container for unknown (dynamic) fields
        self._extra_args: Dict[str, Any] = {}

        # 1) Apply dataclass defaults (if this instance is from a @dataclass subclass)
        if hasattr(self, "__dataclass_fields__"):
            for f in self.__dataclass_fields__.values():
                # If a default is provided, set it
                if f.default is not MISSING:
                    setattr(self, f.name, f.default)
                # If a default_factory is provided, call it and set result
                else:
                    default_factory = getattr(f, "default_factory", MISSING)
                    if default_factory is not MISSING:
                        setattr(self, f.name, default_factory())

        # Prepare set of defined dataclass field names (for distinguishing extras)
        defined_names = set(self.__dataclass_fields__.keys()) if hasattr(self, "__dataclass_fields__") else set()

        # 2) Override / add from kwargs (YAML)
        for k, v in kwargs.items():
            setattr(self, k, v)
            if k not in defined_names:
                # keep track of dynamic/extra args separately as well
                self._extra_args[k] = v

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert all attributes (static dataclass fields + dynamic attrs) to dictionary.
        Priority: use asdict() for dataclass fields when possible, then add extras.
        """
        result: Dict[str, Any] = {}

        # Try to get canonical dataclass representation first
        try:
            result = asdict(self)  # type: ignore
        except Exception:
            # Fallback: collect declared dataclass fields manually (if any)
            if hasattr(self, "__dataclass_fields__"):
                for name in self.__dataclass_fields__.keys():
                    # Use getattr to allow defaults/applied values
                    result[name] = getattr(self, name)

        # Merge dynamic attributes that are not part of asdict result
        for k, v in self.__dict__.items():
            # Skip private/internal attributes
            if k.startswith("_"):
                continue
            if k not in result:
                result[k] = v

        # Ensure extras tracked in _extra_args are present (in case someone set private attr)
        for k, v in getattr(self, "_extra_args", {}).items():
            result.setdefault(k, v)

        return result

    @property
    def extra_args(self) -> Dict[str, Any]:
        """Read-only view of extra (unknown) args provided at init time."""
        return dict(getattr(self, "_extra_args", {}))
    
@dataclass
class ExperimentConfig:
    """Experiment metadata configuration."""
    name: str = "unnamed_experiment"
    description: str = ""
    version: str = "1.0"


@dataclass(init=False)
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

@dataclass(init=False)
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
    add_method: Literal["none", "0_CoT", "FewShot", "FewShot_CoT", "PaS"] = "none" # none for vanilla RE2, CoT for RE2+CoT, PaS for RE2+PaS

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
            output=OutputConfig(**config_dict.get('output', {}))
        )
    
    def to_dict(self) -> dict:
        """Convert Config to dictionary."""
        # Custom to_dict to handle FlexibleConfig properly
        d = asdict(self)
        # Re-inject dynamic fields from model config
        d['model']['init_args'] = self.model.init_args.to_dict()
        d['model']['generation_args'] = self.model.generation_args.to_dict()
        return d