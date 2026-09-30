# =====================================================================
# v5 → thêm cuda-nvcc vào runtime
#        → cho phép JIT compile CUDA trên GPU mới (SM120)
# v6 → thêm libcurand-dev-13-0
# v7 → bỏ flash-attn và lm-evaluation-harness khỏi build-time
#        → chuyển sang cài khi runtime khi cần
# v8 → vllm 0.21.0 → 0.23.0
#        → thêm support/compatibility tốt hơn cho Gemma 4
# v9 → pin llama.cpp:
#        ARG LLAMA_CPP_REF=v0.4.0
#        git clone --branch ${LLAMA_CPP_REF} --depth 1 ...
#        → đảm bảo build đúng revision thay vì lấy branch moving target
# v10 → FlashInfer JIT:
#        lỗi nvcc bị kill, exit code 137
#        → giới hạn Ninja bằng MAX_JOBS
# v11 → bổ sung:
#        cuda-nvrtc-dev-13-0
#        → sửa lỗi thiếu nvrtc.h
#        và
#        CMAKE_CUDA_ARCHITECTURES=...
#        → sửa việc Docker builder không có GPU nên native không detect được architecture
# v12: bật GGML_CUDA_FA_ALL_QUANTS và GGML_CUDA_GRAPHS cho llama.cpp
#      để tăng cường CUDA Flash Attention kernels và CUDA Graph support
# =====================================================================

# =======================
# STAGE 1: BUILDER — chỉ compile llama-server (cần nvcc/cmake)
# =======================
FROM nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04 AS llama-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

ARG LLAMA_CPP_REF=v0.4.0

# Lưu ý: build number vẫn có thể hiển thị "1" dù pin tag, vì --depth 1 vẫn là shallow
# clone (git rev-list --count HEAD luôn = 1 khi thiếu full history). Đây chỉ là vấn đề
# hiển thị, không ảnh hưởng đến việc binary chạy đúng commit nào — commit hash trong
# `--version` vẫn chính xác. Cái quan trọng được fix ở đây là PIN TAG, không phải build number.
ARG CUDA_ARCHITECTURES="80;86;89;90;120"

RUN git clone --branch ${LLAMA_CPP_REF} --depth 1 https://github.com/ggml-org/llama.cpp.git && \
    cd llama.cpp && \
    cmake -B build \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_NO_VMM=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DGGML_CUDA_GRAPHS=ON && \
    cmake --build build --config Release -j"$(nproc)" --target llama-server && \
    find build/bin -maxdepth 1 -type f \( -name "llama-server" -o -name "*.so*" \) \
        -exec strip --strip-unneeded {} \; ; \
    mkdir -p /out && \
    find build/bin -maxdepth 1 -type f \( -name "llama-server" -o -name "*.so*" \) \
        -exec cp {} /out/ \;

# =======================
# STAGE 2: RUNTIME
# =======================
FROM nvidia/cuda:13.0.3-cudnn-runtime-ubuntu24.04 AS final

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    VENV_PATH=/opt/venv \
    HF_HOME=/workspace/.cache/huggingface \
    # Include sm_120/12.0 cho Blackwell (RTX 5060 Ti) cùng các GPU phổ biến
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget python3 python3-venv python3-dev ca-certificates libgomp1 \
    cuda-nvcc-13-0 \
    cuda-nvrtc-dev-13-0 \
    libcurand-dev-13-0 \
    libcublas-dev-13-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:${PATH}"
RUN pip install --upgrade pip

WORKDIR /workspace

# llama-server binary từ builder
COPY --from=llama-builder /out/ /opt/llama.cpp/bin/
RUN ln -s /opt/llama.cpp/bin/llama-server /usr/local/bin/llama-server
ENV LD_LIBRARY_PATH="/opt/llama.cpp/bin:${LD_LIBRARY_PATH}"

# ---- Torch (cu130 index) ----
RUN pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu130

# ---- requirements.txt ----
COPY requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt


WORKDIR /workspace
CMD ["/bin/bash"]