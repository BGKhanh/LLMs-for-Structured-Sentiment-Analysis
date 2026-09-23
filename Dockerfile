# =====================================================================
# Dockerfile — môi trường lm-eval + vLLM + llama.cpp
# v5: thêm cuda-nvcc vào runtime để JIT compile được cho GPU mới
#     (RTX 5060 Ti Blackwell sm_120f), fix editable install path issue
# v6: bỏ cài sẵn flash-attn và lm-evaluation-harness ở build-time
#     → giảm dung lượng image và thời gian build; cài tại runtime khi cần
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
RUN git clone --branch ${LLAMA_CPP_REF} --depth 1 https://github.com/ggml-org/llama.cpp.git && \
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


WORKDIR /workspace
CMD ["/bin/bash"]