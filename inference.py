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
        validate_config(config)
        print("✅ Configuration validated successfully!\n")
        
        # Show config summary
        print("📋 Configuration Summary:")
        print(f"  Experiment : {config.experiment.name}")
        print(f"  Model      : {config.model.name} ({config.model.model_id})")
        print(f"  Technique  : {config.prompt.technique}")
        print(f"  Language   : {'English' if config.prompt.language == 'en' else 'Vietnamese'}")
        print(f"  Dataset    : {config.data.dataset}")
        if config.data.n_sample > 0:
            print(f"  N-samples  : {config.data.n_sample}")
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
    
    
    


# Bước 1: Tạo config file
# Option A: Dùng configuration.py (interactive)
# python configuration.py
# Trả lời các câu hỏi
# File sẽ được tạo tự động

# Option B: Tạo thủ công
# Tạo file YAML trong configs/experiments/ theo template trên. Xem trong ./src/configuration.py

# Bước 2: Run inference
# python inference.py --config configs/experiments/exp_test_rereading.yaml