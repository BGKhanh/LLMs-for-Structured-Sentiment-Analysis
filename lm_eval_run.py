"""
Wrapper script for running Vietnamese SSA evaluation via lm-evaluation-harness.

Usage:
    python lm_eval_run.py \
        --model vllm \
        --model_args "pretrained=google/gemma-3-4b-it,dtype=bfloat16,gpu_memory_utilization=0.9" \
        --technique few_shot --language vi --n_shot 3 \
        --dataset dev --limit 10

    python lm_eval_run.py \
        --model hf \
        --model_args "pretrained=google/gemma-3-4b-it,dtype=bfloat16" \
        --technique rereading --language vi \
        --dataset dev --batch_size 8
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Vietnamese SSA evaluation via lm-evaluation-harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model
    parser.add_argument("--model", type=str, default="vllm", choices=["hf", "vllm"])
    parser.add_argument("--model_args", type=str, required=True)
    parser.add_argument("--batch_size", type=str, default="auto")

    # Prompt technique
    parser.add_argument(
        "--technique",
        type=str,
        required=True,
        choices=["few_shot", "few_shot_cot", "rereading", "plan_and_solve", "re2_pas_cot"],
    )
    parser.add_argument("--language", type=str, default="vi", choices=["vi", "en"])
    parser.add_argument("--n_shot", type=int, default=0)
    parser.add_argument("--plus_mode", action="store_true", help="PS+ mode for plan_and_solve")
    parser.add_argument(
        "--add_method",
        type=str,
        default="none",
        choices=["none", "0_CoT", "FewShot", "FewShot_CoT", "PaS"],
    )
    parser.add_argument("--examples_pool_path", type=str, default=None)

    # Data
    parser.add_argument("--dataset", type=str, default="dev", choices=["dev", "test", "train"])
    parser.add_argument("--dataset_dir", type=str, default="data/vitoed_new")
    parser.add_argument("--limit", type=int, default=None)

    # Output
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--experiment_name", type=str, default=None)

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def _extract_model_name_from_args(model_args: str) -> str:
    """Extract short model name from model_args string for directory naming."""
    for part in model_args.split(","):
        if part.startswith("pretrained="):
            model_id = part.split("=", 1)[1]
            return model_id.split("/")[-1]
    return "unknown_model"


def _build_experiment_name(args) -> str:
    """Build experiment name from arguments."""
    if args.experiment_name:
        return args.experiment_name
    parts = [f"lm_eval_{args.technique}"]
    if args.n_shot > 0:
        parts.append(f"{args.n_shot}shot")
    parts.append(args.language)
    parts.append(args.dataset)
    return "_".join(parts)


def _save_json(path: Path, data, label: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  Saved {label}: {path.name}")


def main():
    args = parse_args()

    # ── Header ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  VIETNAMESE SSA EVALUATION (lm-evaluation-harness)")
    print("=" * 80)
    print(f"  Model     : {args.model} | {args.model_args}")
    print(f"  Technique : {args.technique} (lang={args.language}, n_shot={args.n_shot})")
    print(f"  Dataset   : {args.dataset} ({args.dataset_dir})")
    print(f"  Limit     : {args.limit or 'ALL'}")
    print(f"  Seed      : {args.seed}")
    print("=" * 80 + "\n")

    # ── 1. Set seed BEFORE configure ────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── 2. Configure prompt template ────────────────────────────────────
    # Import task utils (adds project root to sys.path as side effect)
    task_utils_path = (
        Path(__file__).resolve().parent
        / "lm-evaluation-harness"
        / "lm_eval"
        / "tasks"
        / "vietnamese_ssa"
    )
    sys.path.insert(0, str(task_utils_path))

    from lm_eval.tasks.vietnamese_ssa import utils as task_utils

    print("[1/4] Configuring prompt template...")
    system_prompt = task_utils.configure(
        technique=args.technique,
        language=args.language,
        n_shot=args.n_shot,
        plus_mode=args.plus_mode,
        add_method=args.add_method,
        examples_pool_path=args.examples_pool_path,
        dataset_dir=args.dataset_dir,
    )
    print(f"  System prompt length: {len(system_prompt)} chars\n")

    # ── 3. Run evaluation ───────────────────────────────────────────────
    print("[2/4] Running lm-evaluation-harness...")
    import lm_eval

    start_time = time.time()
    results = lm_eval.simple_evaluate(
        model=args.model,
        model_args=args.model_args,
        tasks=["vietnamese_ssa"],
        batch_size=int(args.batch_size) if args.batch_size.isdigit() else args.batch_size,
        system_instruction=system_prompt,
        apply_chat_template=True,
        fewshot_as_multiturn=True,
        log_samples=True,
        limit=args.limit,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        confirm_run_unsafe_code=True,
    )
    elapsed = time.time() - start_time
    print(f"\n  Evaluation completed in {elapsed:.1f}s\n")

    if results is None:
        print("  No results returned (not rank 0). Exiting.")
        return

    # ── 4. Extract and display metrics ──────────────────────────────────
    print("[3/4] Extracting metrics...")
    task_results = results.get("results", {}).get("vietnamese_ssa", {})

    print("\n" + "-" * 40)
    print(f"  {'METRIC':<20} {'SCORE':>10}")
    print("-" * 40)
    for key, val in task_results.items():
        if isinstance(val, (int, float)):
            print(f"  {key:<20} {val:>10.4f}")
    print("-" * 40 + "\n")

    # ── 5. Save results in 4 files ──────────────────────────────────────
    print("[4/4] Saving results...")
    model_short = _extract_model_name_from_args(args.model_args)
    exp_name = _build_experiment_name(args)
    exp_dir = Path(args.output_dir) / model_short / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # -- result.json: SemEval-format predictions
    samples = results.get("samples", {}).get("vietnamese_ssa", [])
    semeval_results = []
    for sample in samples:
        doc = sample.get("doc", {})
        filtered = sample.get("filtered_resps", [])
        resp_text = filtered[0] if filtered else "{}"
        try:
            pred = json.loads(resp_text)
        except (json.JSONDecodeError, TypeError):
            pred = {"opinions": []}
        semeval_results.append(
            {
                "sent_id": doc.get("sent_id"),
                "text": doc.get("text", ""),
                "opinions": pred.get("opinions", []),
            }
        )
    _save_json(exp_dir / "result.json", semeval_results, "Results")

    # -- config.json: full configuration
    config_data = {
        "framework": "lm-evaluation-harness",
        "model": args.model,
        "model_args": args.model_args,
        "technique": args.technique,
        "language": args.language,
        "n_shot": args.n_shot,
        "plus_mode": args.plus_mode,
        "add_method": args.add_method,
        "dataset": args.dataset,
        "dataset_dir": args.dataset_dir,
        "limit": args.limit,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "lm_eval_config": results.get("config", {}),
    }
    _save_json(exp_dir / "config.json", config_data, "Config")

    # -- metadata.json: timing, hardware, metrics
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "experiment_name": exp_name,
        "model_name": model_short,
        "elapsed_time_seconds": round(elapsed, 2),
        "num_samples": len(samples),
        "metrics": {k: v for k, v in task_results.items() if isinstance(v, (int, float))},
    }
    try:
        from src.utils.save_result import _build_metadata

        class _MinimalConfig:
            """Minimal shim so _build_metadata can read experiment/model names."""

            class experiment:
                name = exp_name

            class model:
                name = model_short

        env_meta = _build_metadata(_MinimalConfig, semeval_results, None)
        metadata["environment"] = env_meta.get("environment", {})
    except Exception:
        pass
    _save_json(exp_dir / "metadata.json", metadata, "Metadata")

    # -- debug_info.json: per-sample prompts and raw responses
    debug_data = {
        "technique": args.technique,
        "is_multi_stage": False,
        "shared_prompts": {"system_prompt": system_prompt},
        "samples": [],
    }
    for sample in samples:
        doc = sample.get("doc", {})
        arguments = sample.get("arguments", [[]])
        resps = sample.get("resps", [[]])
        debug_data["samples"].append(
            {
                "sent_id": doc.get("sent_id"),
                "user_prompt": arguments[0][0] if arguments and arguments[0] else "",
                "raw_response": resps[0][0] if resps and resps[0] else "",
            }
        )
    _save_json(exp_dir / "debug_info.json", debug_data, "Debug info")

    print(f"\n  All files saved to: {exp_dir}")
    print("\n" + "=" * 80)
    print("  EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
