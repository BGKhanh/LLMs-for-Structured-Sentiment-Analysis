"""Error analysis CLI — implements ssa_error_analysis_taxonomy_v2.md.

Cách dùng:
    python analyze_errors.py <results_root> [--tau 0.5] [--out report.csv]

<results_root> có thể là 1 thư mục kết quả của 1 kịch bản (chứa
samples_*.jsonl) hoặc thư mục cha chứa nhiều kịch bản (như
OUTPUT_ROOT trong run_all_scenarios.py) — script tự tìm đệ quy.

Mỗi samples_*.jsonl tìm được sẽ có 1 report CSV riêng
(<tên_file>.error_report.csv), cộng với 1 report tổng hợp tất cả sample
gộp lại (--out, mặc định error_report_overall.csv).
"""
import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.error_analysis.aggregate import build_report, write_report_csv
from src.error_analysis.classify import classify_sample
from src.error_analysis.record import load_samples_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", help="Thư mục chứa samples_*.jsonl (đệ quy)")
    parser.add_argument("--tau", type=float, default=0.5, help="Ngưỡng near-miss/merge/split (§7.0), mặc định 0.5")
    parser.add_argument("--out", default="error_report_overall.csv", help="File CSV tổng hợp")
    args = parser.parse_args()

    root = Path(args.results_root)
    jsonl_files = sorted(root.rglob("samples_*.jsonl"))
    if not jsonl_files:
        print(f"Không tìm thấy samples_*.jsonl nào dưới {root}")
        return

    all_samples, all_analyses = [], []
    for path in jsonl_files:
        print(f"--- {path} ---")
        samples = load_samples_jsonl(str(path))
        analyses = [classify_sample(s, tau=args.tau) for s in samples]

        per_file_report = build_report(samples, analyses)
        out_path = path.with_suffix(".error_report.csv")
        write_report_csv(per_file_report, str(out_path))
        print(f"  {len(samples)} samples -> {out_path}")

        all_samples.extend(samples)
        all_analyses.extend(analyses)

    overall = build_report(all_samples, all_analyses)
    write_report_csv(overall, args.out)
    print(f"\nTổng hợp {len(all_samples)} samples từ {len(jsonl_files)} file -> {args.out}")


if __name__ == "__main__":
    main()
