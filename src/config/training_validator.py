# src/config/training_validator.py

"""
Training configuration validation.

Validates training config values before starting fine-tuning.
"""

from pathlib import Path
from typing import List, Tuple
from .training_schema import TrainingConfig


class TrainingConfigValidator:
    """Validate training configuration."""
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize validator.
        
        Args:
            config: TrainingConfig to validate
        """
        self.config = config
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate(self) -> Tuple[bool, List[str], List[str]]:
        """
        Validate all configuration sections.
        
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self._validate_experiment()
        self._validate_model()
        self._validate_data()
        self._validate_prompt()
        self._validate_lora()
        self._validate_training()
        
        is_valid = len(self.errors) == 0
        return is_valid, self.errors, self.warnings
    
    def _validate_experiment(self):
        """Validate experiment metadata."""
        if not self.config.experiment_name:
            self.warnings.append("Experiment name is empty")
        
        if not self.config.experiment_name.replace("_", "").replace("-", "").isalnum():
            self.warnings.append(
                "Experiment name contains special characters (may cause issues)"
            )
    
    def _validate_model(self):
        """Validate model configuration."""
        # Check model path/ID
        if not self.config.model.model_name_or_path:
            self.errors.append("model_name_or_path is required")
        
        # Check quantization conflicts
        if self.config.model.load_in_4bit and self.config.model.load_in_8bit:
            self.errors.append("Cannot use both load_in_4bit and load_in_8bit")
        
        # Warn about memory requirements
        if not (self.config.model.load_in_4bit or self.config.model.load_in_8bit):
            self.warnings.append(
                "No quantization enabled. Training will require significant GPU memory. "
                "Consider setting load_in_4bit=true (QLoRA)"
            )
    
    def _validate_data(self):
        """Validate data configuration."""
        # Check train dataset exists
        train_path = Path(self.config.data.train_dataset_path)
        if not train_path.exists():
            self.errors.append(f"Training dataset not found: {train_path}")
        
        # Check eval dataset (if specified)
        if self.config.data.eval_dataset_path:
            eval_path = Path(self.config.data.eval_dataset_path)
            if not eval_path.exists():
                self.warnings.append(
                    f"Evaluation dataset not found: {eval_path}. "
                    "Training will proceed without evaluation."
                )
        
        # Check max_seq_length
        if self.config.data.max_seq_length < 128:
            self.warnings.append("Very small max_seq_length may truncate data")
        
        if self.config.data.max_seq_length > 4096:
            self.warnings.append(
                f"Large max_seq_length ({self.config.data.max_seq_length}) "
                "will increase memory usage significantly"
            )
    
    def _validate_prompt(self):
        """Validate prompt configuration (reused from inference)."""
        technique = self.config.prompt.technique
        
        # Validate few-shot techniques
        if technique in ["few_shot", "few_shot_cot"]:
            if self.config.prompt.n_shot < 0:
                self.errors.append("n_shot must be >= 0")
            
            if self.config.prompt.n_shot > 10:
                self.warnings.append(
                    f"Large n_shot ({self.config.prompt.n_shot}) "
                    "will make training sequences very long"
                )
            
            # Check examples pool exists (if n_shot > 0)
            if self.config.prompt.n_shot > 0:
                if self.config.prompt.examples_pool_path:
                    pool_path = Path(self.config.prompt.examples_pool_path)
                    if not pool_path.exists():
                        self.errors.append(
                            f"Examples pool file not found: {pool_path}"
                        )
                else:
                    self.errors.append(
                        "examples_pool_path is required when n_shot > 0"
                    )
    
    def _validate_lora(self):
        """Validate LoRA configuration."""
        # Check LoRA rank
        if self.config.lora.r < 1:
            self.errors.append("LoRA rank (r) must be >= 1")
        
        if self.config.lora.r > 64:
            self.warnings.append(
                f"Large LoRA rank ({self.config.lora.r}) will increase "
                "trainable parameters and memory usage"
            )
        
        # Check lora_alpha
        if self.config.lora.lora_alpha < 1:
            self.errors.append("lora_alpha must be >= 1")
        
        # Check dropout
        if not (0.0 <= self.config.lora.lora_dropout < 1.0):
            self.errors.append("lora_dropout must be in [0.0, 1.0)")
        
        # Check target_modules
        if not self.config.lora.target_modules:
            self.errors.append("target_modules cannot be empty")
        
        # Validate QLoRA settings
        if self.config.lora.use_qlora:
            if not self.config.model.load_in_4bit:
                self.warnings.append(
                    "use_qlora=true but load_in_4bit=false. "
                    "QLoRA requires 4-bit quantization. "
                    "Consider setting model.load_in_4bit=true"
                )
    
    def _validate_training(self):
        """Validate training arguments."""
        # Check epochs/steps
        if self.config.training.num_train_epochs < 1 and self.config.training.max_steps < 1:
            self.errors.append("Either num_train_epochs or max_steps must be > 0")
        
        # Check batch sizes
        if self.config.training.per_device_train_batch_size < 1:
            self.errors.append("per_device_train_batch_size must be >= 1")
        
        # Check gradient accumulation
        if self.config.training.gradient_accumulation_steps < 1:
            self.errors.append("gradient_accumulation_steps must be >= 1")
        
        # Effective batch size info
        effective_bs = (
            self.config.training.per_device_train_batch_size * 
            self.config.training.gradient_accumulation_steps
        )
        if effective_bs < 8:
            self.warnings.append(
                f"Small effective batch size ({effective_bs}). "
                "Consider increasing gradient_accumulation_steps"
            )
        
        # Check learning rate
        if self.config.training.learning_rate <= 0:
            self.errors.append("learning_rate must be > 0")
        
        if self.config.training.learning_rate > 1e-3:
            self.warnings.append(
                f"Large learning_rate ({self.config.training.learning_rate}). "
                "Typical range for fine-tuning: 1e-5 to 5e-4"
            )
        
        # Check output dir
        output_dir = Path(self.config.training.output_dir)
        if output_dir.exists():
            self.warnings.append(
                f"Output directory already exists: {output_dir}. "
                "Existing files may be overwritten."
            )
        
        # Check precision settings
        if self.config.training.bf16 and self.config.training.fp16:
            self.errors.append("Cannot use both bf16 and fp16")


def validate_training_config(config: TrainingConfig, verbose: bool = True) -> bool:
    """
    Validate training configuration and print results.
    
    Args:
        config: TrainingConfig to validate
        verbose: Whether to print validation results
        
    Returns:
        True if valid, False otherwise
    
    Raises:
        ValueError: If config is invalid and verbose=False
    """
    validator = TrainingConfigValidator(config)
    is_valid, errors, warnings = validator.validate()
    
    if verbose:
        if warnings:
            print("\n⚠️  Warnings:")
            for warning in warnings:
                print(f"  - {warning}")
        
        if errors:
            print("\n❌ Validation Errors:")
            for error in errors:
                print(f"  - {error}")
        
        if is_valid and not warnings:
            print("✅ Training configuration is valid!")
    
    if not is_valid and not verbose:
        raise ValueError(f"Invalid training configuration: {errors}")
    
    return is_valid