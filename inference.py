# inference.py

"""
Inference CLI entry point.

Usage:
    python inference.py --config configs/experiments/exp_rereading.yaml
    python inference.py -c configs/experiments/exp_few_shot_3.yaml
"""

import argparse
import sys
from pathlib import Path

from src.config import load_config, validate_config
from src.pipeline import InferencePipeline


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with specified configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with experiment config
  python inference.py --config configs/experiments/exp_rereading.yaml
  
  # Short form
  python inference.py -c configs/experiments/exp_few_shot_3.yaml
  
  # Validate config only
  python inference.py -c configs/experiments/test.yaml --validate-only
        """
    )
    
    parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Path to experiment config YAML file"
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate config without running inference"
    )
    
    return parser.parse_args()


def main():
    """Main inference entry point."""
    # Parse arguments
    args = parse_args()
    
    # Print header
    print("\n" + "=" * 80)
    print("🚀 STRUCTURED SENTIMENT ANALYSIS - INFERENCE")
    print("=" * 80)
    print(f"\n📄 Config file: {args.config}\n")
    
    # Check if config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)
    
    try:
        # Load and validate config
        print("🔍 Loading and validating configuration...")
        config = load_config(str(config_path))
        is_valid = validate_config(config)
        
        if not is_valid:
            print("\n❌ Configuration validation failed!")
            sys.exit(1)
        
        print("\n✅ Configuration validated successfully!")
        
        # Show config summary
        print("\n📋 Configuration Summary:")
        print("-" * 80)
        print(f"  Experiment : {config.experiment.name}")
        print(f"  Description: {config.experiment.description}")
        print(f"  Model      : {config.model.name}")
        print(f"  Model ID   : {config.model.init_args.model_id}")
        print(f"  Technique  : {config.prompt.technique}")
        print(f"  Language   : {'English' if config.prompt.language == 'en' else 'Vietnamese'}")
        print(f"  Dataset    : {config.data.dataset}")
        
        # ✅ FIX: Sử dụng num_samples thay vì n_sample
        if config.data.num_samples is not None:
            print(f"  N-samples  : {config.data.num_samples}")
        else:
            print(f"  N-samples  : ALL")
        
        print(f"  Batch size : {config.data.batch_size}")
        
        # Show few-shot info if applicable
        if config.prompt.technique in ["few_shot", "few_shot_cot"]:
            print(f"  N-shot     : {config.prompt.n_shot}")
            if config.prompt.n_shot > 0:
                print(f"  Examples   : {config.data.examples_pool}")
        
        # Show plan-solve mode if applicable
        if config.prompt.technique == "plan_solve":
            mode = "PS+" if config.prompt.plus_mode else "PS"
            print(f"  Mode       : {mode}")
        
        print("-" * 80)
        print()
        
        # If validate-only mode, exit here
        if args.validate_only:
            print("✅ Validation complete! (--validate-only mode)")
            sys.exit(0)
        
        # Initialize and execute pipeline
        print("🔧 Initializing inference pipeline...\n")
        pipeline = InferencePipeline(config=config)
        
        # Execute complete pipeline
        pipeline.execute()
        
        print("\n" + "=" * 80)
        print("✅ INFERENCE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Inference interrupted by user!")
        sys.exit(130)
        
    except Exception as e:
        print(f"\n\n❌ INFERENCE FAILED!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()