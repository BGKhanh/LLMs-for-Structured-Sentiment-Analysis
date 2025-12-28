# src/config/training_schema.py

"""
Training configuration schema using dataclasses.

Extends existing config system for SFT (Supervised Fine-Tuning).
Reuses PromptConfig for consistency and reproducibility.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Literal, List
from .schema import FlexibleConfig, PromptConfig


@dataclass
class LoRAConfig:
    """LoRA/QLoRA configuration for parameter-efficient fine-tuning."""
    # === LoRA Parameters ===
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    
    # === Target Modules ===
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])
    
    # === QLoRA (4-bit quantization) ===
    use_qlora: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass(init=False)
class TrainingArguments(FlexibleConfig):
    """
    Training arguments compatible with HF Trainer.
    
    Uses FlexibleConfig to allow extra arguments (e.g., report_to, hub_model_id).
    """
    # === Output ===
    output_dir: str = "results/training/gemma_sft_v1"
    
    # === Training Duration ===
    num_train_epochs: int = 3
    max_steps: int = -1
    
    # === Batch Size & Gradient ===
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    
    # === Optimizer ===
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_32bit"
    
    # === Logging & Saving ===
    logging_steps: int = 10
    save_strategy: str = "epoch"
    save_total_limit: int = 2
    evaluation_strategy: str = "no"
    
    # === Precision ===
    bf16: bool = True
    fp16: bool = False
    max_grad_norm: float = 0.3
    
    # Extra params handled by FlexibleConfig.__init__


@dataclass
class DataConfig:
    """Data configuration for training."""
    # === Dataset Paths ===
    train_dataset_path: str = "data/vitoed_new/train.json"
    eval_dataset_path: Optional[str] = "data/vitoed_new/dev.json"
    
    # === Processing ===
    max_seq_length: int = 2048
    packing: bool = False


@dataclass
class ModelConfig:
    """Model loading configuration."""
    model_name_or_path: str = "google/gemma-3-4b-it"
    model_type: Literal["gemma", "qwen", "llama", "seallm", "vistral", "vinallama"] = "gemma"
    
    # === Quantization ===
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    
    # === Other ===
    trust_remote_code: bool = True
    cache_dir: Optional[str] = None
    token: Optional[str] = None


@dataclass
class TrainingConfig:
    """
    Complete training configuration.
    
    Reuses PromptConfig from inference schema for consistency.
    """
    # === Experiment Metadata ===
    experiment_name: str = "gemma_sft_baseline"
    random_seed: int = 42
    
    # === Sub-configs ===
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)  # REUSE!
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingArguments = field(default_factory=TrainingArguments)
    
    def __post_init__(self):
        """Ensure sub-configs are converted to objects if they are dicts."""
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)
        if isinstance(self.prompt, dict):
            # Import here to avoid circular dependency
            from .schema import PromptConfig as PC
            self.prompt = PC(**self.prompt)
        if isinstance(self.lora, dict):
            self.lora = LoRAConfig(**self.lora)
        if isinstance(self.training, dict):
            self.training = TrainingArguments(**self.training)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'TrainingConfig':
        """
        Create TrainingConfig from dictionary (YAML).
        
        Args:
            config_dict: Dictionary from YAML
            
        Returns:
            TrainingConfig object
        """
        return cls(
            experiment_name=config_dict.get('experiment_name', 'gemma_sft_baseline'),
            random_seed=config_dict.get('random_seed', 42),
            model=ModelConfig(**config_dict.get('model', {})),
            data=DataConfig(**config_dict.get('data', {})),
            prompt=config_dict.get('prompt', {}),  # Will be converted in __post_init__
            lora=LoRAConfig(**config_dict.get('lora', {})),
            training=TrainingArguments(**config_dict.get('training', {}))
        )
    
    def to_dict(self) -> dict:
        """Convert TrainingConfig to dictionary."""
        d = asdict(self)
        # Handle FlexibleConfig (TrainingArguments)
        if hasattr(self.training, 'to_dict'):
            d['training'] = self.training.to_dict()
        return d