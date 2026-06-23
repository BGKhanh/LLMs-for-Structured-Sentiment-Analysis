# =====================================================================
# Dockerfile — môi trường lm-eval + vLLM + llama.cpp
# v5: thêm cuda-nvcc vào runtime để JIT compile được cho GPU mới
#     (RTX 5060 Ti Blackwell sm_120f), fix editable install path issue
# =====================================================================

# =======================
# STAGE 1: BUILDER — chỉ compile llama-server (cần nvcc/cmake)
# =======================
FROM nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04 AS llama-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp.git && \
    cd llama.cpp && \
    cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_NO_VMM=ON \
                   -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release && \
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
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;12.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget python3 python3-venv python3-dev ca-certificates libgomp1 \
    # cuda-nvcc-13-0: chỉ nvcc + ptxas, không có headers/static libs của devel.
    # Cần thiết vì FlashInfer/vLLM phải JIT compile cho GPU mới (Blackwell sm_120)
    # mà chưa có prebuilt cubin — nhỏ hơn nhiều so với full devel toolkit.
    cuda-nvcc-13-0 libcurand-dev-13-0 \
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

# ---- flash-attn từ wheel dựng sẵn ----
RUN pip install \
    https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn-2.8.3+cu130torch2.11-cp312-cp312-linux_x86_64.whl

# ---- lm-evaluation-harness — NON-editable để tránh path issue ----
# Bản trước dùng `pip install -e .` (editable): venv lưu đường dẫn tuyệt đối
# của source code. Khi copy venv sang stage khác hoặc thay đổi thư mục,
# đường dẫn đó sẽ không còn đúng → import lỗi.
# Non-editable install thì mọi file được copy thẳng vào site-packages,
# không phụ thuộc vào vị trí source code.
RUN git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness.git && \
    cd lm-evaluation-harness && \
    pip install -q ."[hf,api,vllm]"

WORKDIR /workspace
CMD ["/bin/bash"]