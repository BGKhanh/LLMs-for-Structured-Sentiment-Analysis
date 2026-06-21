# =====================================================================
# Dockerfile — môi trường lm-eval + vLLM + llama.cpp (multi-stage để giảm size)
# =====================================================================

# =======================
# STAGE 1: BUILDER — chỉ dùng để compile llama.cpp, bị loại bỏ hoàn toàn
# khỏi image cuối, nên nặng bao nhiêu cũng không ảnh hưởng size push/pull.
# =======================
FROM nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04 AS llama-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# GGML_CUDA_NO_VMM=ON: tránh lỗi link libcuda.so lúc build (không có driver
# thật trong build container).
# BUILD_SHARED_LIBS=OFF: gộp ggml/llama vào tĩnh trong executable, giảm số
# lượng .so cần mang theo sang stage runtime.
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
# STAGE 2: RUNTIME — image thật sự được push/pull, nhẹ hơn nhiều vì không
# mang theo nvcc/headers/static-libs/cmake/build-essential.
# =======================
FROM nvidia/cuda:13.0.3-cudnn-runtime-ubuntu24.04 AS final

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    VENV_PATH=/opt/venv \
    HF_HOME=/workspace/.cache/huggingface

# Chỉ cài những gì THẬT SỰ cần lúc chạy: python venv + git (cho pip install -e)
# + curl/wget (debug/health-check). KHÔNG cài cmake/build-essential nữa.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget python3 python3-venv python3-dev ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:${PATH}"
RUN pip install --upgrade pip

WORKDIR /workspace

# Copy toàn bộ thư mục /out từ builder — gồm binary + bất kỳ .so nào nó cần
# (llama-server-impl khai báo add_library() không ghi rõ STATIC/SHARED, có
# thể vẫn ra .so tuỳ phiên bản CMake/llama.cpp). Copy cả thư mục để chắc
# chắn không bỏ sót, dù BUILD_SHARED_LIBS=OFF có loại bỏ hết .so hay không.
COPY --from=llama-builder /out/ /opt/llama.cpp/bin/
RUN ln -s /opt/llama.cpp/bin/llama-server /usr/local/bin/llama-server
ENV LD_LIBRARY_PATH="/opt/llama.cpp/bin:${LD_LIBRARY_PATH}"

# ---- Torch khớp đúng CUDA 13.0 ----
RUN pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu130

# ---- requirements.txt còn lại ----
COPY requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt

# ---- flash-attn từ wheel dựng sẵn (không cần compiler) ----
RUN pip install \
    https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn-2.8.3+cu130torch2.11-cp312-cp312-linux_x86_64.whl

# ---- lm-evaluation-harness editable + extras ----
RUN git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness.git && \
    cd lm-evaluation-harness && \
    pip install -qe ."[hf,api,vllm]"
# ⚠️ Check sau build: docker run --rm <image> pip show vllm
# (extras có thể tự đổi version vllm khác 0.21.0 đã pin)

WORKDIR /workspace
CMD ["/bin/bash"]