# inference.py

"""
Inference CLI entry point (Minimalist & Accelerated).

Usage:
    accelerate launch inference.py configs/experiments/exp_rereading.yaml
"""

import sys
import traceback
from pathlib import Path
from accelerate import Accelerator, InitProcessGroupKwargs
from datetime import timedelta

from src.config import load_config, validate_config
from src.pipeline import InferencePipeline


def main():
    """Main inference entry point."""
    
    accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=24))
    accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
    
    # 2. Kiểm tra tham số đầu vào
    if len(sys.argv) < 2:
        accelerator.print("❌ Error: Missing config file path.")
        accelerator.print("Usage: accelerate launch inference.py path/to/config.yaml")
        sys.exit(1)
    
    config_path_str = sys.argv[1]
    config_path = Path(config_path_str)

    # 3. Setup Logging
    accelerator.print("\n" + "=" * 60)
    accelerator.print("🚀 INFERENCE START (Accelerated)")
    accelerator.print("=" * 60)
    accelerator.print(f"📄 Config: {config_path}")
    accelerator.print(f"Devices: {accelerator.num_processes}")
    accelerator.print(f"Rank: {accelerator.process_index}")
                      
    # 4. Kiểm tra file tồn tại
    if not config_path.exists():
        accelerator.print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    try:
        # 5. Load Config
        if accelerator.is_local_main_process:
            accelerator.print("🔍 Loading configuration...")
            
        config = load_config(str(config_path))
        
        # (Tùy chọn) Validate config nếu cần thiết, nhưng module config đã lo rồi
        # is_valid = validate_config(config)
        
        # Show config summary (Chỉ in trên Main Process)
        if accelerator.is_local_main_process:
            print("\n📋 Configuration Summary:")
            print("-" * 80)
            print(f"  Experiment : {config.experiment.name}")
            print(f"  Model      : {config.model.name}")
            print(f"  Technique  : {config.prompt.technique}")
            print(f"  Dataset    : {config.data.dataset}")
            print(f"  Batch size : {config.data.batch_size}")
            print("-" * 80)
            print()
        
        # 6. Khởi tạo Pipeline
        accelerator.print("🔧 Initializing inference pipeline...\n")
        
        # ⚠️ QUAN TRỌNG: Phải truyền accelerator vào đây
        pipeline = InferencePipeline(config=config, accelerator=accelerator)
        
        # 7. Chạy
        pipeline.execute()
        
        accelerator.print("\n" + "=" * 80)
        accelerator.print("✅ INFERENCE COMPLETED SUCCESSFULLY!")
        accelerator.print("=" * 80)
        
    except KeyboardInterrupt:
        accelerator.print("\n\n⚠️  Inference interrupted by user!")
        sys.exit(130)
        
    except Exception as e:
        accelerator.print(f"\n\n❌ INFERENCE FAILED on process {accelerator.process_index}!")
        accelerator.print(f"Error: {e}")
        # Chỉ in traceback chi tiết ở process chính
        if accelerator.is_local_main_process:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()