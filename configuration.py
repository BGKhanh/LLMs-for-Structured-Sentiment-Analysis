# configuration.py

"""
Configuration management CLI tool.

Commands:
  - create: Create a new experiment config
  - validate: Validate an existing config
  - show: Display a config with resolved values
"""

import argparse
import yaml
from pathlib import Path
from src.config import load_config, validate_config


def create_config(args):
    """Create a new experiment config interactively."""
    print("=" * 60)
    print("CREATE NEW EXPERIMENT CONFIG")
    print("=" * 60)
    
    # Get experiment details
    print("\n📝 Experiment Details:")
    name = input("Experiment name: ").strip() or "my_experiment"
    description = input("Description: ").strip() or ""
    
    # Select dataset
    print("\n📊 Dataset Selection:")
    print("  1. train")
    print("  2. dev")
    print("  3. test")
    dataset_choice = input("Choose dataset (1-3) [2]: ").strip() or "2"
    datasets = ["train", "dev", "test"]
    dataset = datasets[int(dataset_choice) - 1]
    
    # Get prompt technique
    print("\n🎯 Prompt Technique:")
    print("  1. rereading")
    print("  2. few_shot")
    print("  3. few_shot_cot")
    print("  4. zero_shot_cot")
    print("  5. plan_solve")
    choice = input("Choose technique (1-5) [1]: ").strip() or "1"
    
    techniques = ["rereading", "few_shot", "few_shot_cot", "zero_shot_cot", "plan_solve"]
    technique = techniques[int(choice) - 1]
    
    # Build config dict
    config_dict = {
        'experiment': {
            'name': name,
            'description': description,
            'version': '1.0'
        },
        'data': {
            'dataset': dataset,  # ✅ Set dataset selector
            'n_sample': None
        },
        'prompt': {
            'technique': technique,
            'language': 'vi'
        },
        'output': {
            'output_file': f"{name}_results.json",
            'overwrite': True
        }
    }
    
    # Technique-specific parameters
    if technique in ['few_shot', 'few_shot_cot']:
        n_shot = input(f"\nNumber of examples (n_shot) [3]: ").strip() or "3"
        config_dict['prompt']['n_shot'] = int(n_shot)
        
        if int(n_shot) > 0:
            # Set examples pool
            print("\n📚 Examples Pool:")
            print("  1. train (recommended)")
            print("  2. dev")
            print("  3. test")
            pool_choice = input("Choose examples pool (1-3) [1]: ").strip() or "1"
            examples_pools = ["train", "dev", "test"]
            examples_pool = examples_pools[int(pool_choice) - 1]
            
            config_dict['data']['examples_pool'] = examples_pool  # ✅ Set in data section
    
    elif technique == 'plan_solve':
        plus = input("\nUse PS+ mode? (y/n) [n]: ").strip().lower() == 'y'
        config_dict['prompt']['plus_mode'] = plus
    
    # Limit samples?
    limit = input("\nLimit number of samples? (blank for all): ").strip()
    if limit:
        config_dict['data']['n_sample'] = int(limit)  # ✅ Use n_sample
    
    # Save config
    output_path = Path(f"configs/experiments/{name}.yaml")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving config to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print("✅ Config created successfully!")
    print(f"\nTo validate:")
    print(f"  python configuration.py validate --config {output_path}")
    print(f"\nTo run inference:")
    print(f"  python inference.py --config {output_path}")

def validate_config_cmd(args):
    """Validate a config file."""
    print("=" * 60)
    print("VALIDATE CONFIG")
    print("=" * 60)
    
    try:
        config = load_config(args.config)
        is_valid = validate_config(config, verbose=True)
        
        if is_valid:
            print("\n✅ Config is valid and ready to use!")
            return 0
        else:
            print("\n❌ Config validation failed!")
            return 1
    
    except Exception as e:
        print(f"\n❌ Error loading config: {e}")
        return 1


def show_config(args):
    """Show config with all resolved values."""
    print("=" * 60)
    print("SHOW CONFIG")
    print("=" * 60)
    
    try:
        config = load_config(args.config)
        
        print("\n📄 Resolved Configuration:")
        print(yaml.dump(config.to_dict(), default_flow_style=False, allow_unicode=True))
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Configuration management tool"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create new experiment config')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate a config file')
    validate_parser.add_argument('--config', required=True, help='Path to config file')
    
    # Show command
    show_parser = subparsers.add_parser('show', help='Show resolved config')
    show_parser.add_argument('--config', required=True, help='Path to config file')
    
    args = parser.parse_args()
    
    if args.command == 'create':
        create_config(args)
    elif args.command == 'validate':
        validate_config_cmd(args)
    elif args.command == 'show':
        show_config(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
    
    
    
# # Create new config interactively
# python configuration.py create

# # Validate config
# python configuration.py validate --config configs/experiments/exp_rereading.yaml

# # Show resolved config
# python configuration.py show --config configs/experiments/exp_rereading.yaml