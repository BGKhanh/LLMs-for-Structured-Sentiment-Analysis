"""host.py — CHỈ nạp config server rồi khởi chạy + giữ tiến trình sống.
Độc lập với run.py (evaluation) — đúng workflow hiện tại của bạn: 1 cell/terminal
chạy host.py giữ server sống, cell/terminal khác chạy run.py trỏ base_url tới đây.

    python host.py --config configs/host_vllm_gemma.yaml
    python host.py --config configs/host_llama_qwen.yaml
"""
import argparse
import time

from server_host import host, load_server_config


def main() -> None:
    ap = argparse.ArgumentParser(description="Khởi chạy 1 server suy luận local (vLLM hoặc llama-server) theo config YAML.")
    ap.add_argument("--config", required=True, help="Đường dẫn file YAML cấu hình server.")
    args = ap.parse_args()

    cfg = load_server_config(args.config)
    handle = host(cfg)
    handle.start()
    print(f"\nServer đang chạy tại {cfg.base_url} — nhấn Ctrl+C để tắt.")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        handle.stop()


if __name__ == "__main__":
    main()
