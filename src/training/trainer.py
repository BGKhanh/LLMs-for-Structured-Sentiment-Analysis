# src/training/trainer.py

"""
SFT Trainer wrapper for structured sentiment analysis.

Integrates TRL's SFTTrainer with PEFT (LoRA/QLoRA) for efficient fine-tuning.
Uses existing project config system and follows project standards.
"""

import torch
import os
import json
from pathlib import Path
from typing import Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments as HFTrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

from ..config.training_schema import TrainingConfig


class SentimentSFTTrainer:
    """
    Wrapper for SFTTrainer to fine-tune LLMs on structured sentiment analysis.
    
    Features:
    - Supports LoRA and QLoRA (4-bit/8-bit quantization)
    - Integrates with existing config system
    - Follows project error handling patterns
    - Saves model, adapters, and config
    
    Example:
        >>> config = load_training_config("configs/training/sft_gemma.yaml")
        >>> trainer = SentimentSFTTrainer(config)
        >>> trainer.setup()
        >>> trainer.train(train_dataset, eval_dataset)
        >>> trainer.save()
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize SFT trainer.
        
        Args:
            config: TrainingConfig object with all parameters
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.is_setup = False
        
        print("\n" + "="*70)
        print("🚀 SentimentSFTTrainer Initialized")
        print(f"📝 Experiment: {config.experiment_name}")
        print("="*70)
    
    def setup(self) -> None:
        """
        Setup model, tokenizer, and PEFT configuration.
        
        Steps:
        1. Load tokenizer
        2. Load model (with optional quantization)
        3. Apply PEFT (LoRA)
        
        Raises:
            RuntimeError: If setup fails
        """
        if self.is_setup:
            print("⚠️  Already setup, skipping...")
            return
        
        try:
            print("\n" + "="*70)
            print("🔧 Starting Setup")
            print("="*70)
            
            self._load_tokenizer()
            self._load_model()
            self._apply_peft()
            
            self.is_setup = True
            print("\n✅ Setup completed successfully!\n")
            
        except Exception as e:
            print(f"\n❌ Setup failed: {str(e)}")
            raise RuntimeError(f"Failed to setup trainer: {e}")
    
    def _load_tokenizer(self) -> None:
        """Load and configure tokenizer."""
        print("\n📝 Loading tokenizer...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model.model_name_or_path,
                trust_remote_code=self.config.model.trust_remote_code,
                cache_dir=self.config.model.cache_dir,
                token=self.config.model.token
            )
            
            # Configure padding for training
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print("   ℹ️  Set pad_token = eos_token")
            
            self.tokenizer.padding_side = "right"  # Critical for training!
            
            print(f"✅ Tokenizer loaded: {self.config.model.model_name_or_path}")
            print(f"   Vocab size: {len(self.tokenizer)}")
            print(f"   Padding side: {self.tokenizer.padding_side}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer: {e}")
    
    def _load_model(self) -> None:
        """Load model with optional quantization."""
        print("\n🤖 Loading model...")
        
        try:
            # Prepare quantization config
            quantization_config = None
            
            if self.config.lora.use_qlora or self.config.model.load_in_4bit:
                print("   🔧 Configuring QLoRA (4-bit quantization)...")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=getattr(
                        torch, 
                        self.config.lora.bnb_4bit_compute_dtype
                    ),
                    bnb_4bit_quant_type=self.config.lora.bnb_4bit_quant_type,
                    bnb_4bit_use_double_quant=self.config.lora.bnb_4bit_use_double_quant
                )
                print(f"   ✅ QLoRA config: {self.config.lora.bnb_4bit_quant_type}")
                
            elif self.config.model.load_in_8bit:
                print("   🔧 Configuring 8-bit quantization...")
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                print("   ✅ 8-bit quantization enabled")
                
            model_kwargs = {
                "quantization_config": quantization_config,
                "device_map": "auto",
                "trust_remote_code": self.config.model.trust_remote_code,
                "cache_dir": self.config.model.cache_dir,
                "token": self.config.model.token,
                "torch_dtype": torch.bfloat16 if self.config.training.bf16 else torch.float16
            }
            
            # ✅ ADD: Flash Attention support
            if hasattr(self.config.model, 'attn_implementation') and self.config.model.attn_implementation:
                model_kwargs["attn_implementation"] = self.config.model.attn_implementation
                print(f"   ⚡ Attention: {self.config.model.attn_implementation}")
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model.model_name_or_path,
                **model_kwargs
            )
            
            # Prepare for k-bit training if quantized
            if quantization_config is not None:
                print("   🔧 Preparing model for k-bit training...")
                self.model = prepare_model_for_kbit_training(self.model)
            
            # Enable gradient checkpointing if specified
            if self.config.training.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
                print("   ✅ Gradient checkpointing enabled")
            
            print(f"✅ Model loaded successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def _apply_peft(self) -> None:
        """Apply PEFT (LoRA) configuration to model."""
        print("\n🔧 Applying LoRA...")
        
        try:
            peft_config = LoraConfig(
                r=self.config.lora.r,
                lora_alpha=self.config.lora.lora_alpha,
                lora_dropout=self.config.lora.lora_dropout,
                bias=self.config.lora.bias,
                task_type=self.config.lora.task_type,
                target_modules=self.config.lora.target_modules
            )
            
            print(f"   LoRA rank: {self.config.lora.r}")
            print(f"   LoRA alpha: {self.config.lora.lora_alpha}")
            print(f"   Target modules: {self.config.lora.target_modules}")
            
            self.model = get_peft_model(self.model, peft_config)
            
            # Print trainable parameters
            self.model.print_trainable_parameters()
            print("✅ LoRA applied successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to apply LoRA: {e}")
    
    def train(
        self,
        train_dataset,
        eval_dataset: Optional = None
    ) -> None:
        """
        Start training.
        
        Args:
            train_dataset: Training dataset (PyTorch Dataset or HF Dataset)
            eval_dataset: Optional evaluation dataset
            
        Raises:
            RuntimeError: If training fails
            ValueError: If not setup or invalid dataset
        """
        if not self.is_setup:
            raise ValueError("Must call setup() before train()")
        
        if train_dataset is None or len(train_dataset) == 0:
            raise ValueError("train_dataset is empty or None")
        
        print("\n" + "="*70)
        print("🎯 Starting Training")
        print("="*70)
        print(f"Training samples: {len(train_dataset)}")
        if eval_dataset:
            print(f"Evaluation samples: {len(eval_dataset)}")
        print()
        
        try:
            # Convert TrainingConfig to HF TrainingArguments
            training_args = HFTrainingArguments(
                output_dir=self.config.training.output_dir,
                num_train_epochs=self.config.training.num_train_epochs,
                max_steps=self.config.training.max_steps,
                per_device_train_batch_size=self.config.training.per_device_train_batch_size,
                per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
                gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
                gradient_checkpointing=self.config.training.gradient_checkpointing,
                learning_rate=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                warmup_ratio=self.config.training.warmup_ratio,
                lr_scheduler_type=self.config.training.lr_scheduler_type,
                logging_steps=self.config.training.logging_steps,
                save_strategy=self.config.training.save_strategy,
                save_total_limit=self.config.training.save_total_limit,
                evaluation_strategy=self.config.training.evaluation_strategy,
                bf16=self.config.training.bf16,
                fp16=self.config.training.fp16,
                max_grad_norm=self.config.training.max_grad_norm,
                optim=self.config.training.optim,
                report_to="none",  # Disable wandb/tensorboard by default
                seed=self.config.random_seed,
                # Add extra args from FlexibleConfig if any
                **self.config.training.extra_args
            )
            
            # Initialize SFTTrainer
            print("🔧 Initializing SFTTrainer...")
            self.trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                dataset_text_field="text",
                max_seq_length=self.config.data.max_seq_length,
                packing=self.config.data.packing
            )
            
            print("✅ Trainer initialized\n")
            
            # Start training
            print("🚀 Training started...\n")
            self.trainer.train()
            
            print("\n" + "="*70)
            print("✅ Training completed successfully!")
            print("="*70)
            
        except Exception as e:
            print(f"\n❌ Training failed: {str(e)}")
            raise RuntimeError(f"Training failed: {e}")
    
    def save(self, output_dir: Optional[str] = None) -> None:
        """
        Save fine-tuned model, adapters, and configuration.
        
        Args:
            output_dir: Output directory (default: config.training.output_dir)
            
        Saves:
        - LoRA adapters (adapter_model.safetensors, adapter_config.json)
        - Tokenizer
        - Training config (training_config.json)
        """
        if output_dir is None:
            output_dir = self.config.training.output_dir
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*70)
        print(f"💾 Saving model to: {output_path}")
        print("="*70)
        
        try:
            # Save LoRA adapters
            print("\n📦 Saving LoRA adapters...")
            self.model.save_pretrained(output_path)
            print("✅ LoRA adapters saved")
            
            # Save tokenizer
            print("\n📝 Saving tokenizer...")
            self.tokenizer.save_pretrained(output_path)
            print("✅ Tokenizer saved")
            
            # Save training config
            print("\n⚙️  Saving training config...")
            config_path = output_path / "training_config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config.to_dict(), f, indent=2, ensure_ascii=False)
            print("✅ Training config saved")
            
            print("\n" + "="*70)
            print("✅ All files saved successfully!")
            print("="*70)
            print(f"\n📂 Output directory: {output_path.absolute()}")
            print("\nSaved files:")
            print("  - adapter_model.safetensors  (LoRA weights)")
            print("  - adapter_config.json        (LoRA config)")
            print("  - tokenizer files            (tokenizer_config.json, etc.)")
            print("  - training_config.json       (training configuration)")
            
        except Exception as e:
            print(f"\n❌ Save failed: {str(e)}")
            raise RuntimeError(f"Failed to save model: {e}")
    
    def cleanup(self) -> None:
        """
        Cleanup GPU memory.
        
        Call this after training to free GPU memory.
        """
        print("\n🧹 Cleaning up GPU memory...")
        
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.trainer is not None:
            del self.trainer
            self.trainer = None
        
        torch.cuda.empty_cache()
        
        print("✅ Cleanup completed")