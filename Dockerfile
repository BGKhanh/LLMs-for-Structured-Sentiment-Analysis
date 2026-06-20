# =====================================================================
# Dockerfile — môi trường lm-eval + vLLM + llama.cpp dựng sẵn cho vast.ai
# Build 1 lần, push Docker Hub, dùng lại cho mọi instance.
# =====================================================================

# ⚠️ TỰ KIỂM TRA tag này còn tồn tại trên Docker Hub trước khi build:
#    https://hub.docker.com/r/nvidia/cuda/tags  (tìm "13.0" + "devel" + "ubuntu24.04")
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    VENV_PATH=/opt/venv

# ---- 1. System deps ----
# Ubuntu 24.04 đã có sẵn python3.12 mặc định -> khớp đúng cp312 của flash-attn wheel.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget cmake build-essential \
    python3 python3-venv python3-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---- 2. Tạo venv riêng (tránh PEP668 "externally-managed-environment" của
# Ubuntu 24.04, và tránh kiểu lệch sys.executable vs `pip` đã từng gặp) ----
RUN python3 -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:${PATH}"
RUN pip install --upgrade pip

WORKDIR /workspace

# ---- 3. Build llama.cpp (CUDA) ----
# GGML_CUDA_NO_VMM=ON: lúc `docker build` KHÔNG có GPU/driver thật được pass
# vào container, nên libcuda.so.1 (driver thật) không tồn tại. Mặc định ggml-cuda
# link vào CUDA::cuda_driver (libcuda.so) để dùng VMM allocator -> link fail
# (undefined reference cuMemCreate...). Tắt VMM để chỉ dùng CUDA Runtime API
# (libcudart, luôn có sẵn trong image *-devel*, không cần driver thật lúc build).
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp.git && \
    cd llama.cpp && \
    cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_NO_VMM=ON -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --config Release -j"$(nproc)" && \
    cp build/bin/llama-server /usr/local/bin/llama-server

# ---- 4. Cài torch khớp đúng CUDA 13.0 trước (flash-attn wheel build dựa
# trên đúng combo cu130 + torch2.11, cần khớp tuyệt đối ABI) ----
# ⚠️ Tự xác nhận index-url "cu130" tồn tại trên download.pytorch.org trước khi build.
RUN pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu130

# ---- 5. Copy + cài requirements.txt ----
# torch/torchvision/torchaudio đã cài đúng bản cu130 ở bước 4. pip chỉ check
# version string (==2.11.0) khớp là sẽ SKIP reinstall, không bị ghi đè bởi
# bản torch khác index từ PyPI mặc định -> an toàn khi cài tiếp phần còn lại.
COPY requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt

# ---- 6. Cài flash-attn từ wheel dựng sẵn (bỏ qua compile from source) ----
RUN pip install \
    https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn-2.8.3+cu130torch2.11-cp312-cp312-linux_x86_64.whl

# ---- 7. Clone + cài lm-evaluation-harness editable với extras ----
RUN git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness.git && \
    cd lm-evaluation-harness && \
    pip install -qe ."[hf,api,vllm]"

# ⚠️ Bước trên có thể kéo theo resolver tự đổi version vllm khác với
# vllm==0.21.0 đã pin ở requirements.txt nếu extras "vllm" của lm-eval-harness
# yêu cầu range khác. Build xong nên check:
#   docker run --rm <image> pip show vllm
# Nếu version bị lệch, thêm `pip install vllm==0.21.0 --no-deps` ngay sau dòng
# pip install -qe trên để ép lại đúng version mong muốn.

# ---- 8. KHÔNG bake model weights vào image — xem ghi chú phần trả lời ----
ENV HF_HOME=/workspace/.cache/huggingface

WORKDIR /workspace
CMD ["/bin/bash"]