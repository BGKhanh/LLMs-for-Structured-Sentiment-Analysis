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
    
    def _validate_model(self):
        """Validate model configuration."""
        # Check max_new_tokens
        if self.config.model.max_new_tokens < 1:
            self.errors.append("max_new_tokens must be >= 1")
        
        if self.config.model.max_new_tokens > 4096:
            self.warnings.append(
                f"Large max_new_tokens ({self.config.model.max_new_tokens}) may be slow"
            )
        
        # Check temperature with do_sample
        if self.config.model.do_sample and not (0.0 <= self.config.model.temperature <= 2.0):
            self.warnings.append(
                f"temperature={self.config.model.temperature} is unusual (typical: 0.1-1.0)"
            )
    
    def _validate_data(self):
        """Validate data configuration."""
        # Check dataset exists
        dataset_path = Path(self.config.data.dataset_path)
        if not dataset_path.exists():
            self.errors.append(f"Dataset file not found: {dataset_path}")
        
        # Check batch_size
        if self.config.data.batch_size < 1:
            self.errors.append("batch_size must be >= 1")
        
        if self.config.data.batch_size > 32:
            self.warnings.append(
                f"Large batch_size ({self.config.data.batch_size}) may cause OOM"
            )
        
        # Check num_samples
        if self.config.data.num_samples is not None and self.config.data.num_samples < 1:
            self.errors.append("num_samples must be >= 1 or null")
    
    def _validate_prompt(self):
        """Validate prompt configuration."""
        technique = self.config.prompt.technique
        
        # Validate few-shot specific parameters
        if technique in ["few_shot", "few_shot_cot"]:
            if self.config.prompt.n_shot < 0:
                self.errors.append("n_shot must be >= 0")
            
            if self.config.prompt.n_shot > 0:
                # examples_pool_path is required
                if not self.config.prompt.examples_pool_path:
                    self.errors.append(
                        f"examples_pool_path required for {technique} with n_shot > 0"
                    )
                else:
                    # Check file exists
                    pool_path = Path(self.config.prompt.examples_pool_path)
                    if not pool_path.exists():
                        self.errors.append(
                            f"Examples pool file not found: {pool_path}"
                        )
            
            if self.config.prompt.n_shot > 5:
                self.warnings.append(
                    f"Large n_shot ({self.config.prompt.n_shot}) may make prompts very long"
                )
    
    def _validate_output(self):
        """Validate output configuration."""
        output_dir = Path(self.config.output.output_dir)
        
        # Check if output dir needs to be created
        if not output_dir.exists():
            self.warnings.append(f"Output directory will be created: {output_dir}")
        
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
            print("\n✅ Config validation passed!")
    
    return is_valid