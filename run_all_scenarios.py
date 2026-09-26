"""
Chạy tự động 11 kịch bản prompt chuẩn (re2, pas, 0/1/3/5/7/10-shot, 1/3/5-shot-cot)
cho MỘT model trên MỘT dataset, không cần tự đổi tên task / --num_fewshot mỗi lần.

Vì sao dùng Python API (simple_evaluate) thay vì gọi `lm_eval` CLI 11 lần trong 1
vòng lặp bash: mỗi lần gọi CLI = 1 lần nạp lại toàn bộ engine vLLM (load weight
model lên GPU) — với model lớn, riêng bước load có thể tốn vài phút, nhân 11 lần
là lãng phí đáng kể. Script này build model 1 LẦN DUY NHẤT rồi tái sử dụng cho
cả 11 lần evaluate.

Cách dùng:
    1. Sửa các biến trong khối CONFIG bên dưới (MODEL_ARGS, DATASET, ...).
    2. Chạy: python run_all_scenarios.py
    3. Kết quả nằm ở OUTPUT_ROOT/<tên_kịch_bản>/ — mỗi kịch bản 1 thư mục riêng,
       đúng định dạng results.json + samples_*.jsonl như khi chạy CLI thông thường.

Không cần sửa gì bên dưới khối CONFIG, trừ khi muốn đổi chính danh sách 11 kịch bản.
"""

import json
import os
from pathlib import Path

from lm_eval import simple_evaluate
from lm_eval.api.registry import get_model
from lm_eval.loggers import EvaluationTracker
from lm_eval.tasks import TaskManager
from lm_eval.utils import handle_non_serializable, make_table

# ============================== CONFIG ==============================
# Model — giữ đúng cú pháp model_args như khi dùng CLI (comma-separated).
MODEL_BACKEND = "vllm"
MODEL_ARGS = "pretrained=google/gemma-4-E2B-it,max_model_len=16384,gpu_memory_utilization=0.95"

# Dataset — dùng tên DATASET (không phải ngôn ngữ), khớp
# `src/prompt_templates/shared.py::DATASET_LANGUAGES`. Ví dụ: vitoed_new,
# opener_en, mpqa, darmstadt_unis, opener_es, norec, multibooked_eu, multibooked_ca.
DATASET = "multibooked_eu"
VERBOSE_DATA = False
INCLUDE_PATH = "./src/tasks"
OUTPUT_ROOT = Path("results/lm_eval") / MODEL_ARGS.split("pretrained=")[1].split(",")[0].replace("/", "__")

BATCH_SIZE = 64
APPLY_CHAT_TEMPLATE = True
LOG_SAMPLES = True
CONFIRM_RUN_UNSAFE_CODE = True

# random_seed, numpy_random_seed, torch_random_seed, fewshot_random_seed
# (tương đương --seed "0,1234,1234,42" ở CLI)
SEEDS = dict(random_seed=0, numpy_random_seed=1234, torch_random_seed=1234, fewshot_random_seed=42)

# (task_name, num_fewshot, thư_mục_output_con) — đúng 11 kịch bản đã mô tả.
# few_shot_cot bị giới hạn bởi kích thước pool tĩnh trong shared.py (hiện tại 5
# ví dụ cho tiếng Việt) — 5-shot-cot là mức tối đa có thể chạy với dataset "vi".
SCENARIOS = [
    ("vietnamese_ssa_re_reading",   0,  "re2"),
    ("vietnamese_ssa_plan_and_solve", 0, "pas"),
    ("vietnamese_ssa_few_shot",     0,  "0_shot"),
    ("vietnamese_ssa_few_shot",     1,  "1_shot"),
    ("vietnamese_ssa_few_shot",     3,  "3_shot"),
    ("vietnamese_ssa_few_shot",     5,  "5_shot"),
    ("vietnamese_ssa_few_shot",     7,  "7_shot"),
    ("vietnamese_ssa_few_shot",     10, "10_shot"),
    ("vietnamese_ssa_few_shot_cot", 1,  "1_shot_cot"),
    ("vietnamese_ssa_few_shot_cot", 3,  "3_shot_cot"),
    ("vietnamese_ssa_few_shot_cot", 5,  "5_shot_cot"),
]
# ======================================================================


def main():
    os.environ.setdefault("PYTHONHASHSEED", "42")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    print(f"[1/3] Đang nạp model ({MODEL_BACKEND}: {MODEL_ARGS}) — chỉ 1 lần cho toàn bộ {len(SCENARIOS)} kịch bản...")
    model_cls = get_model(MODEL_BACKEND)
    lm_obj = model_cls.create_from_arg_string(MODEL_ARGS, {"batch_size": BATCH_SIZE})

    print(f"[2/3] Đang dựng TaskManager (include_path={INCLUDE_PATH}, dataset={DATASET})...")
    task_manager = TaskManager(include_path=INCLUDE_PATH, metadata={"dataset": DATASET,"clean_data_verbose":VERBOSE_DATA})

    print(f"[3/3] Chạy {len(SCENARIOS)} kịch bản, kết quả tại: {OUTPUT_ROOT}/\n")
    for i, (task_name, num_fewshot, subdir) in enumerate(SCENARIOS, 1):
        out_dir = OUTPUT_ROOT / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"--- [{i}/{len(SCENARIOS)}] {subdir}: task={task_name} num_fewshot={num_fewshot} -> {out_dir} ---")

        evaluation_tracker = EvaluationTracker(output_path=str(out_dir))

        results = simple_evaluate(
            model=lm_obj, 
            tasks=[task_name],
            num_fewshot=num_fewshot,
            batch_size=BATCH_SIZE,
            log_samples=LOG_SAMPLES,
            evaluation_tracker=evaluation_tracker,
            apply_chat_template=APPLY_CHAT_TEMPLATE,
            task_manager=task_manager,
            confirm_run_unsafe_code=CONFIRM_RUN_UNSAFE_CODE,
            **SEEDS,
        )

        if results is None:
            print(f"    (không có kết quả trả về cho {subdir} — có thể do rank != 0 trong chạy phân tán)")
            continue

        samples = results.pop("samples") if LOG_SAMPLES else None

        # Ghi results.json / results table — đúng logic evaluation_tracker.save_results_aggregated
        # mà CLI `lm_eval run` dùng nội bộ (xem lm_eval/_cli/run.py).
        evaluation_tracker.save_results_aggregated(results=results, samples=samples if LOG_SAMPLES else None)
        if LOG_SAMPLES and samples:
            for t_name in results["configs"]:
                evaluation_tracker.save_results_samples(task_name=t_name, samples=samples[t_name])

        dumped = json.dumps(results, indent=2, default=handle_non_serializable, ensure_ascii=False)
        print(f"    Xong. {out_dir}/results.json đã ghi ({len(dumped)} bytes).\n")

        # In bảng kết quả (đúng format CLI `lm_eval run` in ra) trước khi sang kịch bản tiếp theo.
        print(f"    === Kết quả [{subdir}] (task={task_name}, num_fewshot={num_fewshot}) ===")
        print(make_table(results))
        if "groups" in results:
            print(make_table(results, "groups"))
        print()

    print(f"\nHoàn tất toàn bộ {len(SCENARIOS)} kịch bản. Kết quả tại: {OUTPUT_ROOT}/")


if __name__ == "__main__":
    main()