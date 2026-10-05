"""run.py — chỉ NẠP config rồi CHẠY. Mọi thiết lập nằm ở file YAML (thư mục configs/).

    python run.py --config configs/gemma_local.yaml
    python run.py --config configs/gemma_local.yaml --dry-run                 # chỉ in kế hoạch, không chạy
    python run.py --config configs/gemma_local.yaml --only re2,3_shot         # chạy lọc vài scenario
    python run.py --config configs/gemma_local.yaml --dataset mpqa            # ghi đè danh sách dataset

Riêng cho `model.mode: hf` (backend "hf" có sẵn trong lm_eval, nạp
`transformers` trực tiếp — xem `configs/hf.yaml`) VÀ muốn chạy
data-parallel nhiều GPU: bọc lệnh trên bằng `accelerate launch`, KHÔNG sửa gì
trong code — run.py là script Python thường, `accelerate launch` tự spawn N
tiến trình (mỗi GPU 1 tiến trình) rồi chạy đúng y nguyên script/args này
trong từng tiến trình:

    accelerate launch run.py --config configs/gemma_hf.yaml
"""
import argparse
import os

from lm_eval_runner import LMEvalRunner, _is_main_process, load_experiment


def main() -> None:
    ap = argparse.ArgumentParser(description="Chạy các kịch bản prompt SSA theo file config YAML.")
    ap.add_argument("--config", required=True, help="Đường dẫn file YAML cấu hình.")
    ap.add_argument("--dataset", help="Ghi đè run.datasets (phân tách bằng dấu phẩy).")
    ap.add_argument("--only", help="Chỉ chạy các scenario có label này (phân tách bằng dấu phẩy).")
    ap.add_argument("--dry-run", action="store_true", help="Chỉ in kế hoạch chạy, không nạp model.")
    args = ap.parse_args()

    exp = load_experiment(args.config)
    datasets = args.dataset.split(",") if args.dataset else exp.run.datasets
    scenarios = exp.scenarios
    if args.only:
        wanted = args.only.split(",")
        unknown = set(wanted) - {s.label for s in scenarios}
        if unknown:
            raise SystemExit(f"--only: label không tồn tại: {sorted(unknown)}. Có: {[s.label for s in scenarios]}")
        scenarios = [s for s in scenarios if s.label in wanted]

    os.environ.setdefault("PYTHONHASHSEED", "42")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    runner = LMEvalRunner(exp.model, exp.run)
    if _is_main_process():
        print(f"Model: backend={exp.model.backend} tag={exp.model.tag} | datasets={datasets} | {len(scenarios)} scenario")
        for ds in datasets:
            for sc in scenarios:
                print(f"  {runner.output_dir(ds, sc)}  <-  {sc.task} (num_fewshot={sc.num_fewshot})")
    if args.dry_run:
        return
    runner.run_all(scenarios, datasets)


if __name__ == "__main__":
    main()