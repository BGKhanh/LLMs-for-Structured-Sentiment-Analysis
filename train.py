# train.py

"""
Training CLI entry point for fine-tuning LLMs on structured sentiment analysis.

Usage:
    python train.py --config configs/training/sft_gemma_baseline.yaml
    python train.py -c configs/training/sft_gemma_qlora.yaml
"""

import argparse
import sys
from pathlib import Path

from src.config import load_training_config, validate_training_config
from src.training import SentimentSFTTrainer, prepare_training_datasets
from src.utils.random_seed import set_seed

# Import prompt templates
from src.prompt_templates import (
    FewShotPrompt,
    ZeroShotCoTPrompt,
    FewShotCoTPrompt,
    ReReadingPrompt,
    PlanAndSolvePrompt
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune LLM with specified training configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with LoRA
  python train.py --config configs/training/sft_gemma_baseline.yaml
  
  # Train with QLoRA (4-bit)
  python train.py -c configs/training/sft_gemma_qlora.yaml
  
  # Short form
  python train.py -c configs/training/sft_gemma.yaml
        """
    )
    
    parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Path to training config YAML file"
    )
    
    return parser.parse_args()


def get_prompt_template(config):
    """
    Factory function to create prompt template based on config.
    
    Args:
        config: TrainingConfig object
        
    Returns:
        Initialized prompt template
        
    Note:
        Reuses existing prompt templates for consistency with inference!
    """
    technique = config.prompt.technique
    language = config.prompt.language
    eng = (language == "en")
    
    # Create template based on technique
    if technique == "rereading":
        template = ReReadingPrompt(eng=eng)
        
    elif technique == "few_shot":
        template = FewShotPrompt(
            eng=eng,
            n_shot=config.prompt.n_shot,
            examples_pool_path=config.prompt.examples_pool_path
        )
        
    elif technique == "few_shot_cot":
        template = FewShotCoTPrompt(
            eng=eng,
            n_shot=config.prompt.n_shot,
            examples_pool_path=config.prompt.examples_pool_path
        )
        
    elif technique == "zero_shot_cot":
        template = ZeroShotCoTPrompt(eng=eng)
        
    elif technique == "plan_and_solve":
        template = PlanAndSolvePrompt(
            eng=eng,
            plus_mode=config.prompt.plus_mode
        )
        
    else:
        raise ValueError(f"Unknown prompt technique: {technique}")
    
    # Prepare template (load examples, cache system prompt)
    print(f"🔧 Preparing prompt template: {technique}")
    template.prepare()
    
    return template


def main():
    """Main training entry point."""
    # Parse arguments
    args = parse_args()
    
    # Print header
    print("\n" + "=" * 80)
    print("🚀 STRUCTURED SENTIMENT ANALYSIS - TRAINING")
    print("=" * 80)
    print(f"\n📄 Config file: {args.config}")
    print()
    
    # Check if config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)
    
    try:
        # ========== STEP 1: Load and validate config ==========
        print("🔍 Loading and validating configuration...")
        config = load_training_config(str(config_path))
        is_valid = validate_training_config(config)
        
        if not is_valid:
            print("\n❌ Configuration validation failed!")
            sys.exit(1)
        
        print("\n✅ Configuration validated successfully!")
        
        # ========== STEP 2: Show config summary ==========
        print("\n📋 Configuration Summary:")
        print("-" * 80)
        print(f"  Experiment     : {config.experiment_name}")
        print(f"  Model          : {config.model.model_type}")
        print(f"  Model ID       : {config.model.model_name_or_path}")
        print(f"  Prompt Tech    : {config.prompt.technique}")
        print(f"  Language       : {'English' if config.prompt.language == 'en' else 'Vietnamese'}")
        
        # Show few-shot info if applicable
        if config.prompt.technique in ["few_shot", "few_shot_cot"]:
            print(f"  N-shot         : {config.prompt.n_shot}")
        
        print(f"\n  Training Data  : {config.data.train_dataset_path}")
        if config.data.eval_dataset_path:
            print(f"  Eval Data      : {config.data.eval_dataset_path}")
        
        print(f"\n  LoRA Rank      : {config.lora.r}")
        print(f"  LoRA Alpha     : {config.lora.lora_alpha}")
        print(f"  QLoRA Enabled  : {'Yes' if config.lora.use_qlora or config.model.load_in_4bit else 'No'}")
        
        print(f"\n  Epochs         : {config.training.num_train_epochs}")
        print(f"  Batch Size     : {config.training.per_device_train_batch_size}")
        print(f"  Grad Accum     : {config.training.gradient_accumulation_steps}")
        effective_bs = (config.training.per_device_train_batch_size * 
                       config.training.gradient_accumulation_steps)
        print(f"  Effective BS   : {effective_bs}")
        print(f"  Learning Rate  : {config.training.learning_rate}")
        
        print(f"\n  Output Dir     : {config.training.output_dir}")
        print("-" * 80)
        print()
        
        # ========== STEP 3: Set random seed ==========
        print(f"🎲 Setting random seed: {config.random_seed}")
        set_seed(config.random_seed)
        print()
        
        # ========== STEP 4: Initialize trainer ==========
        trainer = SentimentSFTTrainer(config)
        trainer.setup()
        
        # ========== STEP 5: Prepare prompt template ==========
        print("\n" + "=" * 80)
        print("📝 Preparing Prompt Template")
        print("=" * 80)
        prompt_template = get_prompt_template(config)
        print(f"✅ Prompt template ready: {config.prompt.technique} ({config.prompt.language})")
        print()
        
        # ========== STEP 6: Prepare datasets ==========
        print("\n" + "=" * 80)
        print("📊 Preparing Training Datasets")
        print("=" * 80)
        train_dataset, eval_dataset = prepare_training_datasets(
            train_path=config.data.train_dataset_path,
            eval_path=config.data.eval_dataset_path,
            prompt_generator=prompt_template.get_prompt,
            tokenizer=trainer.tokenizer
        )
        
        print(f"\n✅ Datasets ready:")
        print(f"   Training samples: {len(train_dataset)}")
        if eval_dataset:
            print(f"   Evaluation samples: {len(eval_dataset)}")
        print()
        
        # ========== STEP 7: Train! ==========
        trainer.train(train_dataset, eval_dataset)
        
        # ========== STEP 8: Save model ==========
        trainer.save()
        
        # ========== STEP 9: Cleanup ==========
        trainer.cleanup()
        
        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\n📂 Model saved to: {config.training.output_dir}")
        print("\nNext steps:")
        print("  1. Test the fine-tuned model with inference.py")
        print("  2. Evaluate on test set")
        print("  3. Compare with baseline results")
        print()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        print("💡 Tip: Partial checkpoint may be saved in output directory")
        sys.exit(130)
        
    except Exception as e:
        print(f"\n\n❌ TRAINING FAILED!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
    
    
    
    
    
# # Basic training with LoRA
# python train.py --config configs/training/sft_gemma_baseline.yaml

# # Memory-efficient training with QLoRA
# python train.py -c configs/training/sft_gemma_qlora.yaml

# # With different prompt techniques
# python train.py -c configs/training/sft_gemma_fewshot.yaml