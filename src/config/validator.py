# src/config/validator.py

"""
Configuration validation.

Validates config values and checks for common errors before running inference.
"""

from pathlib import Path
from typing import List, Tuple
from .schema import Config


class ConfigValidator:
    """Validate configuration before inference."""
    
    def __init__(self, config: Config):
        """
        Initialize validator.
        
        Args:
            config: Configuration to validate
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
        self._validate_output()
        
        is_valid = len(self.errors) == 0
        return is_valid, self.errors, self.warnings
    
    def _validate_experiment(self):
        """Validate experiment configuration."""
        if not self.config.experiment.name:
            self.warnings.append("Experiment name is empty")
        
        if not self.config.experiment.name.replace("_", "").replace("-", "").isalnum():
            self.warnings.append(
                "Experiment name contains special characters (may cause issues in file paths)"
            )
    
    def _validate_model(self):
        """Validate model configuration."""
        # Check max_new_tokens
        if self.config.model.max_new_tokens < 1:
            self.errors.append("max_new_tokens must be >= 1")
        
        if self.config.model.max_new_tokens > 8192:
            self.warnings.append(
                f"Very large max_new_tokens ({self.config.model.max_new_tokens}) may be slow"
            )
        
        # Check temperature with do_sample
        if self.config.model.do_sample:
            if not (0.0 <= self.config.model.temperature <= 2.0):
                self.warnings.append(
                    f"temperature={self.config.model.temperature} is unusual (typical: 0.1-1.5)"
                )
            
            if not (0.0 <= self.config.model.top_p <= 1.0):
                self.errors.append("top_p must be between 0.0 and 1.0")
            
            if self.config.model.top_k < 1:
                self.errors.append("top_k must be >= 1")
        
        # Check dtype
        valid_dtypes = ["float32", "float16", "bfloat16", "auto"]
        if self.config.model.dtype not in valid_dtypes:
            self.errors.append(f"dtype must be one of {valid_dtypes}")
        
        # Note: We do NOT validate additional_model_kwargs
        # Advanced users are responsible for correctness
        if self.config.model.additional_model_kwargs:
            self.warnings.append(
                "Using additional_model_kwargs - ensure parameters are correct"
            )
    
    def _validate_data(self):
        """Validate data configuration."""
        # Check dataset file exists (use method instead of direct field)
        try:
            dataset_path = Path(self.config.data.get_dataset_path())
            if not dataset_path.exists():
                self.errors.append(f"Dataset file not found: {dataset_path}")
        except Exception as e:
            self.errors.append(f"Error getting dataset path: {e}")
        
        # Check batch_size
        if self.config.data.batch_size < 1:
            self.errors.append("batch_size must be >= 1")
        
        if self.config.data.batch_size > 32:
            self.warnings.append(
                f"Large batch_size ({self.config.data.batch_size}) may cause OOM"
            )
        
        # Check num_samples
        if self.config.data.num_samples is not None:
            if self.config.data.num_samples < 1:
                self.errors.append("num_samples must be >= 1 or null")
        
        # Check num_workers
        if getattr(self.config.data, "num_workers", 0) < 0:
            self.errors.append("num_workers must be >= 0")
    
    def _validate_prompt(self):
        """Validate prompt configuration."""
        technique = self.config.prompt.technique
        
        # Validate technique-specific parameters
        if technique in ["few_shot", "few_shot_cot"]:
            # Check n_shot
            if self.config.prompt.n_shot < 0:
                self.errors.append("n_shot must be >= 0")
            
            if self.config.prompt.n_shot > 10:
                self.warnings.append(
                    f"Very large n_shot ({self.config.prompt.n_shot}) may make prompts extremely long"
                )
            
            # Check examples pool exists (if n_shot > 0)
            if self.config.prompt.n_shot > 0:
                try:
                    # Get examples pool path from DataConfig
                    pool_path = Path(self.config.data.get_examples_pool_path())
                    if not pool_path.exists():
                        self.errors.append(
                            f"Examples pool file not found: {pool_path}\n"
                            f"  Set data.examples_pool to 'train', 'dev', or 'test'"
                        )
                except Exception as e:
                    self.errors.append(f"Error getting examples pool path: {e}")
        
        elif technique == "plan_solve":
            # Plan-and-Solve has plus_mode parameter
            pass  # plus_mode is bool, no validation needed
        
        elif technique == "zero_shot_cot":
            # Zero-shot CoT has no special parameters to validate
            pass
        
        elif technique == "rereading":
            # Re-reading has no special parameters
            pass
        
        # Validate language
        valid_languages = ["vi", "en"]
        if self.config.prompt.language not in valid_languages:
            self.errors.append(f"language must be one of {valid_languages}")
    
    def _validate_output(self):
        """Validate output configuration."""
        output_dir = Path(self.config.output.output_dir)
        
        # Check if output dir needs to be created
        if not output_dir.exists():
            self.warnings.append(
                f"Output directory will be created: {output_dir}"
            )
        
        # Check if parent directory is writable
        parent_dir = output_dir.parent if output_dir.parent.exists() else Path(".")
        if not parent_dir.exists():
            self.errors.append(
                f"Parent directory does not exist: {parent_dir}"
            )
        
        # Check overwrite setting
        output_file = output_dir / self.config.output.output_file
        if output_file.exists() and not self.config.output.overwrite:
            self.errors.append(
                f"Output file exists and overwrite=false: {output_file}\n"
                f"  Set output.overwrite=true or change output_file"
            )
        
        # Warn if overwriting
        if output_file.exists() and self.config.output.overwrite:
            self.warnings.append(f"Will overwrite existing file: {output_file}")


def validate_config(config: Config, verbose: bool = True) -> bool:
    """
    Validate configuration and print results.
    
    Args:
        config: Configuration to validate
        verbose: Whether to print validation results
        
    Returns:
        True if valid, False otherwise
    
    Raises:
        ValueError: If config is invalid and verbose=False
    """
    validator = ConfigValidator(config)
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
            print("✅ Configuration is valid!")
    
    if not is_valid and not verbose:
        raise ValueError(f"Invalid configuration: {errors}")
    
    return is_valid