"""
lm_eval_runner.py — THƯ VIỆN chạy lm-eval-harness kiểu pythonic, cấu hình
nằm ở file YAML riêng (xem thư mục `configs/`), file chạy (`run.py`) chỉ nạp
config rồi chạy.

Khối chính:
    ModelConfig / ScenarioConfig / RunConfig   dataclass cấu hình
    ExperimentConfig + load_experiment(path)    nạp + kiểm tra YAML (báo lỗi nếu gõ sai khoá)
    LMEvalRunner                                 build model 1 lần, chạy nhiều scenario x dataset
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from lm_eval import simple_evaluate
from lm_eval.api.model import LM
from lm_eval.api.registry import get_model
from lm_eval.loggers import EvaluationTracker
from lm_eval.tasks import TaskManager
from lm_eval.utils import make_table, simple_parse_args_string


def _is_main_process() -> bool:
    """True khi KHÔNG chạy đa tiến trình (bình thường), hoặc khi chạy đa tiến
    trình qua `accelerate launch` (cho backend "hf" đa-GPU data-parallel) VÀ
    đây là tiến trình rank 0. Dùng cùng 2 biến môi trường `accelerate launch`
    thật sự set (`RANK`, `LOCAL_RANK`) — đã verify khớp với chính cách
    `lm_eval/evaluator.py` tự kiểm tra rank nội bộ. Chỉ ảnh hưởng việc IN RA
    CONSOLE (tránh N tiến trình cùng in log trùng lặp) — simple_evaluate() đã
    tự xử lý đúng việc chỉ rank 0 mới thực sự có `results` để ghi file."""
    return int(os.environ.get("RANK", "0")) == 0 and int(os.environ.get("LOCAL_RANK", "0")) == 0


# =========================================================================
# Helpers
# =========================================================================
def _check_keys(section: str, data: dict, allowed: set[str]) -> None:
    """Báo lỗi ngay nếu YAML có khoá lạ — gõ sai tên khoá (vd `num_fewshots`)
    là lỗi im lặng kinh điển của file config, nên chặn từ đầu."""
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"[config] Khoá không hợp lệ trong '{section}': {sorted(unknown)}. Cho phép: {sorted(allowed)}")


def _model_args_to_string(model_args: dict[str, Any]) -> str:
    return ",".join(f"{k}={v}" for k, v in model_args.items())


def infer_model_tag(model_args: dict[str, Any] | str) -> str:
    """Tên model để đặt thư mục output, không phụ thuộc backend (vllm/hf dùng
    `pretrained=`, local-chat-completions dùng `model=`; backend lạ -> fallback an toàn)."""
    parsed = model_args if isinstance(model_args, dict) else simple_parse_args_string(model_args)
    for key in ("pretrained", "model", "engine"):
        if key in parsed and parsed[key]:
            return str(parsed[key]).replace("/", "__").replace(":", "_")
    raw = _model_args_to_string(parsed) if isinstance(model_args, dict) else model_args
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in raw)[:100]


# =========================================================================
# Config dataclasses
# =========================================================================
@dataclass
class ModelConfig:
    backend: str
    args: dict[str, Any]
    batch_size: int | str = 32

    def to_arg_string(self) -> str:
        return _model_args_to_string(self.args)

    @property
    def tag(self) -> str:
        return infer_model_tag(self.args) or self.backend  # args rỗng -> dùng tên backend

    @classmethod
    def vllm(cls, pretrained: str, batch_size: int | str = 32, **extra: Any) -> "ModelConfig":
        """vLLM nạp weight trực tiếp vào tiến trình này (build 1 lần có lợi rõ rệt)."""
        return cls("vllm", {"pretrained": pretrained, **extra}, batch_size)

    @classmethod
    def api_server(cls, model: str, base_url: str, batch_size: int | str = 1, **extra: Any) -> "ModelConfig":
        """Model host sẵn (llama-server / vllm serve), chỉ là HTTP client. batch_size=1
        vì chat request không-tokenized chỉ hỗ trợ 1; song song hoá qua `num_concurrent`.
        Cần: pip install "lm_eval[api]"."""
        return cls("local-chat-completions", {"model": model, "base_url": base_url, **extra}, batch_size)

    @classmethod
    def hf(cls, pretrained: str, batch_size: int | str = "auto", **extra: Any) -> "ModelConfig":
        return cls("hf", {"pretrained": pretrained, **extra}, batch_size)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        _check_keys("model", d, {"mode", "backend", "args", "batch_size"})
        mode, backend = d.get("mode"), d.get("backend")
        if (mode is None) == (backend is None):
            raise ValueError("[config] 'model' cần ĐÚNG MỘT trong các khoá: `mode` (vllm | api_server | hf) hoặc `backend` (tên backend lm-eval bất kỳ).")
        args = dict(d.get("args") or {})
        if mode == "vllm":
            if "pretrained" not in args:
                raise ValueError("[config] model.mode=vllm cần `args.pretrained`.")
            return cls.vllm(batch_size=d.get("batch_size", 32), **args)
        if mode == "api_server":
            missing = {"model", "base_url"} - set(args)
            if missing:
                raise ValueError(f"[config] model.mode=api_server thiếu trong `args`: {sorted(missing)}.")
            return cls.api_server(batch_size=d.get("batch_size", 1), **args)
        if mode == "hf":
            if "pretrained" not in args:
                raise ValueError("[config] model.mode=hf cần `args.pretrained`.")
            return cls.hf(batch_size=d.get("batch_size", "auto"), **args)
        if mode is not None:
            raise ValueError(f"[config] model.mode={mode!r} không hợp lệ (chỉ nhận vllm | api_server | hf).")
        return cls(backend, args, d.get("batch_size", 1))


# Các khoá generation_kwargs PHỔ BIẾN (khớp _common_yaml/few_shot_cot.yaml hiện
# tại) — chỉ để CẢNH BÁO nếu gõ sai chính tả (vd "temprature"), KHÔNG chặn key
# lạ: generation_kwargs được forward thẳng xuống backend (vllm SamplingParams,
# OpenAI-style API, ...) qua dict.update(), nên bất kỳ key nào backend thật sự
# hỗ trợ (vd repetition_penalty, min_p, seed, presence_penalty...) đều hợp lệ
# dù không nằm trong danh sách dưới đây. Khác với `_check_keys` (dùng cho
# khoá CẤU TRÚC của chính ExperimentConfig — những khoá đó luôn là danh sách
# đóng, gõ sai phải chặn cứng).
_KNOWN_GEN_KWARGS_KEYS = {"until", "max_gen_toks", "temperature", "top_p", "top_k", "do_sample"}


def _check_gen_kwargs(source: str, gk: dict[str, Any] | None) -> None:
    if gk:
        unknown = set(gk) - _KNOWN_GEN_KWARGS_KEYS
        if unknown:
            print(
                f"[config] Lưu ý: '{source}.gen_kwargs' có khoá lạ {sorted(unknown)} "
                f"— không nằm trong danh sách phổ biến {sorted(_KNOWN_GEN_KWARGS_KEYS)}. "
                "Vẫn cho qua (có thể là tham số riêng của backend bạn dùng) — tự kiểm "
                "tra lại nếu đây là gõ sai chính tả."
            )


@dataclass
class ScenarioConfig:
    label: str  # cũng là tên thư mục output
    task: str
    num_fewshot: int = 0
    # Override generation_kwargs CHỈ cho scenario này (vd CoT cần max_gen_toks
    # lớn hơn hẳn). Merge kiểu dict.update() vào generation_kwargs sẵn có
    # trong yaml của task — không cần sửa yaml, chỉ ghi đè đúng key chỉ định.
    gen_kwargs: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        _check_gen_kwargs(f"scenarios[{self.label}]", self.gen_kwargs)


def default_ssa_scenarios() -> list[ScenarioConfig]:
    """11 kịch bản chuẩn: re2, pas, 0/1/3/5/7/10-shot, 1/3/5-shot-cot."""
    return [
        ScenarioConfig("re2", "vietnamese_ssa_re_reading", 0),
        ScenarioConfig("pas", "vietnamese_ssa_plan_and_solve", 0),
        *[ScenarioConfig(f"{n}_shot", "vietnamese_ssa_few_shot", n) for n in (0, 1, 3, 5, 7, 10)],
        *[ScenarioConfig(f"{n}_shot_cot", "vietnamese_ssa_few_shot_cot", n) for n in (1, 3, 5)],
    ]


SCENARIO_PRESETS = {"default_ssa": default_ssa_scenarios}


@dataclass
class RunConfig:
    datasets: list[str]
    include_path: str = "./src/tasks"
    output_root: Path | str = Path("results/lm_eval")
    apply_chat_template: bool = True
    log_samples: bool = True
    confirm_run_unsafe_code: bool = True
    random_seed: int = 0
    numpy_random_seed: int = 1234
    torch_random_seed: int = 1234
    fewshot_random_seed: int = 42
    extra_metadata: dict[str, Any] = field(default_factory=dict)
    # Override generation_kwargs cho CẢ LOẠT scenario (vd đổi temperature cho
    # toàn bộ 11 kịch bản cùng lúc). Scenario nào tự khai `gen_kwargs` riêng
    # thì phần riêng đó thắng (merge 2 tầng — xem LMEvalRunner.run()).
    gen_kwargs: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        _check_gen_kwargs("run", self.gen_kwargs)

    def seeds(self) -> dict[str, int]:
        return dict(random_seed=self.random_seed, numpy_random_seed=self.numpy_random_seed,
                    torch_random_seed=self.torch_random_seed, fewshot_random_seed=self.fewshot_random_seed)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunConfig":
        _check_keys("run", d, {"datasets", "include_path", "output_root", "apply_chat_template", "log_samples",
                               "confirm_run_unsafe_code", "seeds", "extra_metadata", "gen_kwargs"})
        ds = d.get("datasets")
        if isinstance(ds, str):
            ds = [ds]
        if not ds:
            raise ValueError("[config] 'run.datasets' bắt buộc (tên dataset hoặc danh sách, vd [vitoed_new, mpqa]).")
        seeds = dict(d.get("seeds") or {})
        _check_keys("run.seeds", seeds, {"random", "numpy", "torch", "fewshot"})
        kw: dict[str, Any] = {k: d[k] for k in ("include_path", "output_root", "apply_chat_template",
                                                "log_samples", "confirm_run_unsafe_code", "gen_kwargs") if k in d}
        for short, full in (("random", "random_seed"), ("numpy", "numpy_random_seed"),
                            ("torch", "torch_random_seed"), ("fewshot", "fewshot_random_seed")):
            if short in seeds:
                kw[full] = seeds[short]
        return cls(datasets=list(ds), extra_metadata=dict(d.get("extra_metadata") or {}), **kw)


@dataclass
class ExperimentConfig:
    model: ModelConfig
    run: RunConfig
    scenarios: list[ScenarioConfig]


def load_experiment(path: str | Path) -> ExperimentConfig:
    """Nạp 1 file YAML thành ExperimentConfig (đã kiểm tra khoá + giá trị bắt buộc)."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    _check_keys("<root>", raw, {"model", "run", "scenarios"})
    for req in ("model", "run", "scenarios"):
        if req not in raw:
            raise ValueError(f"[config] Thiếu mục '{req}' trong {path}.")

    sc = raw["scenarios"]
    if isinstance(sc, str):
        if sc not in SCENARIO_PRESETS:
            raise ValueError(f"[config] scenarios preset {sc!r} không tồn tại. Có: {sorted(SCENARIO_PRESETS)}")
        scenarios = SCENARIO_PRESETS[sc]()
    else:
        scenarios = []
        for i, item in enumerate(sc):
            _check_keys(f"scenarios[{i}]", item, {"label", "task", "num_fewshot", "gen_kwargs"})
            scenarios.append(ScenarioConfig(**item))
    labels = [s.label for s in scenarios]
    if len(labels) != len(set(labels)):
        raise ValueError(f"[config] `label` của scenarios phải duy nhất (mỗi label = 1 thư mục output): {labels}")

    return ExperimentConfig(ModelConfig.from_dict(raw["model"]), RunConfig.from_dict(raw["run"]), scenarios)


# =========================================================================
# Runner
# =========================================================================
class LMEvalRunner:
    """Model build lazy đúng 1 lần cho cả loạt (scenario x dataset); mỗi dataset
    có TaskManager riêng (vì dataset được truyền qua metadata lúc load task)."""

    def __init__(self, model_config: ModelConfig, run_config: RunConfig):
        self.model_config = model_config
        self.run_config = run_config
        self._lm: LM | None = None
        self._task_managers: dict[str, TaskManager] = {}

    @property
    def lm(self) -> LM:
        if self._lm is None:
            cls = get_model(self.model_config.backend)
            self._lm = cls.create_from_arg_string(self.model_config.to_arg_string(),
                                                  {"batch_size": self.model_config.batch_size})
        return self._lm

    def task_manager(self, dataset: str) -> TaskManager:
        if dataset not in self._task_managers:
            self._task_managers[dataset] = TaskManager(
                include_path=self.run_config.include_path,
                metadata={"dataset": dataset, **self.run_config.extra_metadata},
            )
        return self._task_managers[dataset]

    def output_dir(self, dataset: str, scenario: ScenarioConfig) -> Path:
        return Path(self.run_config.output_root) / self.model_config.tag / dataset / scenario.label

    def run(self, scenario: ScenarioConfig, dataset: str, print_table: bool = True) -> dict[str, Any]:
        out_dir = self.output_dir(dataset, scenario)
        out_dir.mkdir(parents=True, exist_ok=True)
        rc = self.run_config

        # Merge 2 tầng: gen_kwargs của run áp dụng chung, scenario tự khai
        # riêng thì đè lên (dict.update() — chỉ đè đúng key chỉ định, không
        # xoá key khác của tầng run).
        gen_kwargs = dict(rc.gen_kwargs or {})
        gen_kwargs.update(scenario.gen_kwargs or {})

        tracker = EvaluationTracker(output_path=str(out_dir))
        results = simple_evaluate(
            model=self.lm, tasks=[scenario.task], num_fewshot=scenario.num_fewshot,
            batch_size=self.model_config.batch_size, log_samples=rc.log_samples,
            evaluation_tracker=tracker, apply_chat_template=rc.apply_chat_template,
            task_manager=self.task_manager(dataset), gen_kwargs=gen_kwargs or None,
            confirm_run_unsafe_code=rc.confirm_run_unsafe_code, **rc.seeds(),
        )
        if results is None:
            return {}
        samples = results.pop("samples") if rc.log_samples else None
        tracker.save_results_aggregated(results=results, samples=samples)
        if rc.log_samples and samples:
            for t in results["configs"]:
                tracker.save_results_samples(task_name=t, samples=samples[t])
        if print_table and _is_main_process():
            print(f"=== [{dataset} / {scenario.label}] task={scenario.task} num_fewshot={scenario.num_fewshot} -> {out_dir} ===")
            print(make_table(results))
        return results

    def run_all(self, scenarios: list[ScenarioConfig], datasets: list[str] | None = None) -> dict[tuple[str, str], dict[str, Any]]:
        """Duyệt dataset (ngoài) x scenario (trong). Trả về {(dataset, label): results}.
        Khi chạy qua `accelerate launch` (nhiều tiến trình), MỌI tiến trình đều
        chạy đúng vòng lặp này (bắt buộc — HFLM cần tất cả rank cùng tham gia
        để chia dữ liệu đồng đều), chỉ có in log ra console là được gate theo
        rank 0 (xem `_is_main_process()`) để tránh N dòng log trùng lặp."""
        datasets = datasets or self.run_config.datasets
        total, out = len(datasets) * len(scenarios), {}
        n = 0
        for ds in datasets:
            for sc in scenarios:
                n += 1
                if _is_main_process():
                    print(f"--- [{n}/{total}] dataset={ds} scenario={sc.label} ---")
                out[(ds, sc.label)] = self.run(sc, ds)
        return out