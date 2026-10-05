"""
server_host.py — Đóng gói việc HOST model local (vLLM openai-server,
llama-server, hoặc `transformers serve`) thành module cấu hình được, cùng
nguyên lý với `lm_eval_runner.py`: dataclass config, nạp/validate từ YAML,
script (`host.py`) chỉ nạp rồi chạy. 3 backend: `backend: vllm` ->
VLLMServerConfig, `backend: llama_cpp` -> LlamaServerConfig, `backend:
transformers` -> TransformersServerConfig.

Lưu ý riêng cho `transformers`: module này chỉ phục vụ trường hợp bạn muốn 1
SERVER SỐNG RIÊNG (giống vllm/llama-server). Nếu chỉ cần nạp model
`transformers` trực tiếp vào tiến trình chạy eval — không cần server riêng —
dùng thẳng backend "hf" có sẵn trong lm_eval qua `lm_eval_runner.py`'s
`ModelConfig(backend="hf", args={"pretrained": ...})`, không liên quan gì
tới file này.

ĐỘC LẬP với lm_eval_runner.py/run.py — dùng đứng riêng:
    - Chạy `python host.py --config configs/host_vllm_gemma.yaml` (hoặc
      `host_llama_qwen.yaml` / `host_transformers_gemma.yaml`) ở 1
      terminal/cell, giữ server sống, rồi ở chỗ khác trỏ `base_url` trong
      config đánh giá (`configs/gemma_api_server.yaml`) tới đúng server này.
    - Hoặc gọi trực tiếp từ code khác (notebook, script khác) qua
      `with host(cfg) as h: ...` — tự start, tự cleanup khi ra khỏi block
      hoặc khi tiến trình Python thoát (atexit), y hệt cơ chế 2 cell notebook
      gốc của bạn (vllm/llama-server), chỉ khác là tham số hoá qua config.
"""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _check_keys(section: str, data: dict, allowed: set[str]) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"[config] Khoá không hợp lệ trong '{section}': {sorted(unknown)}. Cho phép: {sorted(allowed)}")


# =========================================================================
# Config: vLLM — `python -m vllm.entrypoints.openai.api_server`
# =========================================================================
@dataclass
class VLLMServerConfig:
    model: str
    host: str = "127.0.0.1"
    port: int = 8000
    gpu_memory_utilization: float = 0.975
    max_model_len: int = 16384
    tensor_parallel_size: int | None = None
    dtype: str | None = None
    max_num_batched_tokens: int | None = None
    reasoning_parser: str | None = None
    enable_thinking: bool | None = None  # -> --default-chat-template-kwargs '{"enable_thinking": ...}'
    extra_args: list[str] = field(default_factory=list)  # cờ tuỳ ý chưa có field riêng
    env: dict[str, str] = field(default_factory=dict)  # biến môi trường bổ sung (vd MAX_JOBS)
    log_dir: str = "results/lm_eval"
    start_timeout_s: float = 1200.0

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def health_url(self) -> str:
        return self.base_url.rstrip("/") + "/models"

    def build_command(self) -> list[str]:
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model,
            "--host", self.host,
            "--port", str(self.port),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--max-model-len", str(self.max_model_len),
        ]
        if self.tensor_parallel_size:
            cmd += ["--tensor-parallel-size", str(self.tensor_parallel_size)]
        if self.dtype:
            cmd += ["--dtype", self.dtype]
        if self.max_num_batched_tokens:
            cmd += ["--max-num-batched-tokens", str(self.max_num_batched_tokens)]
        if self.reasoning_parser:
            cmd += ["--reasoning-parser", self.reasoning_parser]
        if self.enable_thinking is not None:
            flag = "true" if self.enable_thinking else "false"
            cmd += ["--default-chat-template-kwargs", f'{{"enable_thinking": {flag}}}']
        cmd += self.extra_args
        return cmd

    def build_env(self) -> dict[str, str]:
        # Ghi chú FlashInfer JIT / host-RAM OOM (xem FlashInfer issue #3634):
        # MAX_JOBS quá lớn -> quá nhiều nvcc song song -> OOM lúc JIT-compile
        # kernel NVFP4. Mặc định "1" cho an toàn (giữ đúng default gốc trong
        # notebook của bạn); tăng lên "2"/"4" qua `env:` nếu RAM đủ.
        e = os.environ.copy()
        e.setdefault("MAX_JOBS", "1")
        e.update({k: str(v) for k, v in self.env.items()})
        hf = e.get("HUGGINGFACE_API_KEY") or e.get("HF_TOKEN") or e.get("HUGGINGFACE_HUB_TOKEN") or e.get("HUGGINGFACE_API", "")
        if hf:
            e.setdefault("HF_TOKEN", hf)
            e.setdefault("HUGGINGFACE_HUB_TOKEN", hf)
        return e

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VLLMServerConfig":
        _check_keys("server(vllm)", d, set(cls.__dataclass_fields__) | {"backend"})
        return cls(**{k: v for k, v in d.items() if k != "backend"})


# =========================================================================
# Config: llama-server (llama.cpp)
# =========================================================================
@dataclass
class LlamaServerConfig:
    binary: str = "/usr/local/bin/llama-server"
    hf_repo_id: str | None = None
    hf_filename: str | None = None
    model_path: str | None = None  # nếu đã có sẵn .gguf local, bỏ qua tải HF
    host: str = "127.0.0.1"
    port: int = 8000
    ctx_size: int = 65536
    n_gpu_layers: int = 99
    batch_size: int = 2048
    ubatch_size: int = 512
    parallel: int = 4
    flash_attn: str = "on"
    reasoning: str = "off"  # "on" | "off"
    reasoning_format: str = "auto"
    jinja: bool = True
    cont_batching: bool = True
    kv_unified: bool = True
    extra_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)  # vd GGML_CUDA_*, CUDA_VISIBLE_DEVICES
    log_dir: str = "."
    start_timeout_s: float = 300.0

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def health_url(self) -> str:
        return self.base_url + "/health"

    def build_env(self) -> dict[str, str]:
        e = os.environ.copy()
        e.update({k: str(v) for k, v in self.env.items()})
        return e

    def resolve_model_path(self) -> Path:
        if self.model_path:
            return Path(self.model_path)
        if not (self.hf_repo_id and self.hf_filename):
            raise ValueError("[config] llama_cpp cần `model_path` HOẶC cả `hf_repo_id` + `hf_filename`.")
        from huggingface_hub import hf_hub_download

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
        print(f"⬇️  Đang kiểm tra/tải model từ {self.hf_repo_id} ...")
        path = Path(hf_hub_download(repo_id=self.hf_repo_id, filename=self.hf_filename, local_dir=".", token=token))
        print(f"✅ Model sẵn sàng tại: {path.resolve()}")
        return path

    def build_command(self, model_path: Path) -> list[str]:
        if not Path(self.binary).exists():
            raise FileNotFoundError(
                f"Không tìm thấy llama-server tại: {self.binary}. Hãy build trước bằng cmake."
            )
        cmd = [
            self.binary, "-m", str(model_path),
            "--host", self.host, "--port", str(self.port),
            "--batch-size", str(self.batch_size),
            "--ubatch-size", str(self.ubatch_size),
            "-ngl", str(self.n_gpu_layers),
            "--parallel", str(self.parallel),
            "--ctx-size", str(self.ctx_size),
            "--reasoning", self.reasoning,
            "--reasoning-format", self.reasoning_format,
            "--flash-attn", self.flash_attn,
        ]
        if self.jinja:
            cmd.append("--jinja")
        if self.cont_batching:
            cmd.append("--cont-batching")
        if self.kv_unified:
            cmd.append("--kv-unified")
        cmd += self.extra_args
        return cmd

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LlamaServerConfig":
        _check_keys("server(llama_cpp)", d, set(cls.__dataclass_fields__) | {"backend"})
        return cls(**{k: v for k, v in d.items() if k != "backend"})


# =========================================================================
# Config: transformers serve (`transformers serve [FORCE_MODEL] ...`)
# =========================================================================
# Khác bản chất với vLLM/llama.cpp ở 1 điểm: `transformers` KHÔNG hỗ trợ gọi
# qua `python -m transformers` (không có __main__.py — đã verify: lệnh này
# báo lỗi "transformers is a package and cannot be directly executed"), nên
# phải gọi thẳng binary CLI `transformers` (cài qua pip, nằm trong PATH của
# venv/conda env đang dùng) — giống cách gọi `llama-server` hơn là `vllm`.
#
# Lưu ý quan trọng: đây là backend HOST RIÊNG (server quá trình dài, có HTTP
# API) — khác với backend "hf" có sẵn trong lm_eval (`ModelConfig(backend="hf",
# args={"pretrained": ...})`), vốn nạp model TRỰC TIẾP vào cùng tiến trình
# chạy eval, KHÔNG cần host.py/server_host.py gì cả. Dùng "hf" nếu bạn không
# cần 1 server độc lập; dùng TransformersServerConfig (backend: transformers
# trong file host config) nếu muốn 1 server sống riêng để nhiều lần `run.py`
# cùng kết nối tới, giống hệt cách dùng vllm/llama_cpp.
@dataclass
class TransformersServerConfig:
    model: str  # FORCE_MODEL — model duy nhất được preload, dùng cho mọi request
    binary: str = "transformers"  # đổi thành full path nếu không nằm trong PATH
    host: str = "localhost"
    port: int = 8000
    device: str = "auto"
    dtype: str = "auto"
    attn_implementation: str | None = None  # vd "flash_attention_2"
    continuous_batching: bool = False
    quantization: str | None = None  # "bnb-4bit" | "bnb-8bit"
    reasoning: str = "auto"  # "on" | "off" | "auto"
    chat_template_kwargs: dict[str, Any] | None = None  # vd {"enable_thinking": true}
    trust_remote_code: bool = False
    model_timeout: int | None = None
    cb_block_size: int | None = None
    cb_num_blocks: int | None = None
    cb_max_batch_tokens: int | None = None
    cb_max_memory_percent: float | None = None
    cb_use_cuda_graph: bool = False
    enable_cors: bool = False
    log_level: str | None = None
    extra_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    log_dir: str = "."
    start_timeout_s: float = 600.0

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def health_url(self) -> str:
        return f"http://{self.host}:{self.port}/health"

    def build_command(self) -> list[str]:
        cmd = [
            self.binary, "serve", self.model,
            "--host", self.host, "--port", str(self.port),
            "--device", self.device, "--dtype", self.dtype,
            "--reasoning", self.reasoning,
        ]
        if self.attn_implementation:
            cmd += ["--attn-implementation", self.attn_implementation]
        if self.continuous_batching:
            cmd.append("--continuous-batching")
        if self.quantization:
            cmd += ["--quantization", self.quantization]
        if self.chat_template_kwargs:
            cmd += ["--chat-template-kwargs", json.dumps(self.chat_template_kwargs)]
        if self.trust_remote_code:
            cmd.append("--trust-remote-code")
        if self.model_timeout is not None:
            cmd += ["--model-timeout", str(self.model_timeout)]
        if self.cb_block_size is not None:
            cmd += ["--cb-block-size", str(self.cb_block_size)]
        if self.cb_num_blocks is not None:
            cmd += ["--cb-num-blocks", str(self.cb_num_blocks)]
        if self.cb_max_batch_tokens is not None:
            cmd += ["--cb-max-batch-tokens", str(self.cb_max_batch_tokens)]
        if self.cb_max_memory_percent is not None:
            cmd += ["--cb-max-memory-percent", str(self.cb_max_memory_percent)]
        if self.cb_use_cuda_graph:
            cmd.append("--cb-use-cuda-graph")
        if self.enable_cors:
            cmd.append("--enable-cors")
        if self.log_level:
            cmd += ["--log-level", self.log_level]
        cmd += self.extra_args
        return cmd

    def build_env(self) -> dict[str, str]:
        e = os.environ.copy()
        e.update({k: str(v) for k, v in self.env.items()})
        return e

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TransformersServerConfig":
        _check_keys("server(transformers)", d, set(cls.__dataclass_fields__) | {"backend"})
        return cls(**{k: v for k, v in d.items() if k != "backend"})


ServerConfig = VLLMServerConfig | LlamaServerConfig | TransformersServerConfig


def load_server_config(path: str | Path) -> ServerConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    backend = raw.get("backend")
    if backend == "vllm":
        return VLLMServerConfig.from_dict(raw)
    if backend == "llama_cpp":
        return LlamaServerConfig.from_dict(raw)
    if backend == "transformers":
        return TransformersServerConfig.from_dict(raw)
    raise ValueError(
        f"[config] 'backend' phải là 'vllm', 'llama_cpp' hoặc 'transformers', nhận được: {backend!r}"
    )


# =========================================================================
# Vòng đời tiến trình server — dùng chung cho cả 2 backend
# =========================================================================
class ServerHandle:
    """Quản lý 1 tiến trình server: launch, health-check theo polling, và tự
    dọn dẹp (atexit + `.stop()` tường minh + context manager). Y hệt logic
    2 cell notebook gốc (terminate -> wait(timeout) -> kill nếu cần), chỉ
    gom vào 1 class dùng chung cho cả vLLM lẫn llama-server."""

    def __init__(self, cmd: list[str], health_url: str, log_path: Path,
                 start_timeout_s: float, env: dict[str, str] | None = None, cwd: str | None = None):
        self.cmd = cmd
        self.health_url = health_url
        self.log_path = Path(log_path)
        self.start_timeout_s = start_timeout_s
        self.env = env or os.environ.copy()
        self.cwd = cwd
        self._proc: subprocess.Popen | None = None
        self._flog = None

    def start(self) -> None:
        self.stop()  # dọn tiến trình CỦA CHÍNH OBJECT NÀY nếu gọi lại (self._proc) —
        # KHÔNG dọn được process mồ côi (orphan) từ 1 tiến trình Python KHÁC đã thoát
        # trước đó (vd kernel notebook bị restart cứng, atexit không kịp chạy). Vì vậy
        # cần kiểm tra health_url TRƯỚC khi launch: nếu đã có server trả lời sẵn ở đó,
        # launch tiếp sẽ rất dễ "trông như thành công" trong khi health-check vòng lặp
        # bên dưới thực ra đang thấy server CŨ (cấu hình cũ, nhỏ hơn) chứ không phải
        # tiến trình MỚI vừa Popen() với tham số trong config hiện tại — đây chính là
        # kiểu lỗi "tăng batch_size/ubatch_size/ctx_size trong YAML nhưng VRAM dùng
        # không đổi" vì so với lần trước. Chặn cứng ở đây để không bị lừa.
        try:
            with urllib.request.urlopen(self.health_url, timeout=3) as r:
                if r.status == 200:
                    raise RuntimeError(
                        f"❌ Đã có server TRẢ LỜI SẴN tại {self.health_url} trước khi host.py kịp "
                        "khởi chạy. Đây rất có thể là tiến trình CŨ còn sót lại từ lần chạy trước "
                        "(orphan process — vd kernel notebook bị restart cứng, Ctrl+C không propagate "
                        "kịp) — KHÔNG phải tiến trình mới với tham số trong config hiện tại. Nếu tiếp "
                        "tục, bạn sẽ 'tưởng' server mới đã chạy (log vẫn in '✅ Server sẵn sàng') "
                        "nhưng thực ra vẫn đang nói chuyện với config CŨ. Hãy tự tắt tiến trình đang "
                        "chiếm cổng này trước (vd `pkill -f llama-server` hoặc `lsof -i :<port>` để tìm "
                        "đúng PID), hoặc đổi sang `port` khác trong config, rồi chạy lại."
                    )
        except (urllib.error.URLError, TimeoutError, OSError):
            pass  # không có gì đang chạy ở đó — đúng như kỳ vọng, tiếp tục launch bình thường

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._flog = open(self.log_path, "a", encoding="utf-8", buffering=1)
        self._flog.write(f"\n\n==== start {time.strftime('%Y-%m-%d %H:%M:%S')} ====\n{' '.join(self.cmd)}\n\n")
        self._flog.flush()
        print(f"🚀 Đang khởi chạy server — log: {self.log_path}")
        self._proc = subprocess.Popen(self.cmd, env=self.env, stdout=self._flog, stderr=subprocess.STDOUT, cwd=self.cwd)
        atexit.register(self.stop)
        self._wait_healthy()

    def _wait_healthy(self) -> None:
        deadline = time.time() + self.start_timeout_s
        last_err = None
        print("⏳ Đang chờ server sẵn sàng (model lớn có thể mất vài phút)...")
        while time.time() < deadline:
            if self._proc.poll() is not None:
                tail = self.log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
                raise RuntimeError(f"❌ Server thoát sớm (exit code={self._proc.returncode}). Log: {self.log_path}\n{tail}")
            try:
                with urllib.request.urlopen(self.health_url, timeout=5) as r:
                    if r.status == 200:
                        print(f"✅ Server sẵn sàng: {self.health_url}")
                        return
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last_err = e
            time.sleep(2.0)
        self.stop()
        raise RuntimeError(f"⏳ Server không sẵn sàng kịp thời. Lỗi cuối: {last_err}. Log: {self.log_path}")

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            print("\n[Cleanup] Đang tắt server an toàn...")
            self._proc.terminate()
            try:
                self._proc.wait(timeout=15)
                print("[Cleanup] Đã tắt thành công.")
            except subprocess.TimeoutExpired:
                print("[Cleanup] Không phản hồi, ép buộc tắt (kill)...")
                self._proc.kill()
        if self._flog:
            self._flog.close()
            self._flog = None

    def __enter__(self) -> "ServerHandle":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


def host(config: ServerConfig) -> ServerHandle:
    """Dựng ServerHandle đúng theo loại config — CHƯA start. Gọi `.start()`
    hoặc dùng `with host(cfg) as h:` để tự start + tự cleanup."""
    if isinstance(config, VLLMServerConfig):
        return ServerHandle(
            cmd=config.build_command(), health_url=config.health_url(),
            log_path=Path(config.log_dir) / "vllm_server.log",
            start_timeout_s=config.start_timeout_s, env=config.build_env(),
        )
    if isinstance(config, LlamaServerConfig):
        model_path = config.resolve_model_path()
        return ServerHandle(
            cmd=config.build_command(model_path), health_url=config.health_url(),
            log_path=Path(config.log_dir) / "llama_server.log",
            start_timeout_s=config.start_timeout_s, env=config.build_env(),
        )
    if isinstance(config, TransformersServerConfig):
        return ServerHandle(
            cmd=config.build_command(), health_url=config.health_url(),
            log_path=Path(config.log_dir) / "transformers_server.log",
            start_timeout_s=config.start_timeout_s, env=config.build_env(),
        )
    raise TypeError(f"Config không hợp lệ: {type(config)}")