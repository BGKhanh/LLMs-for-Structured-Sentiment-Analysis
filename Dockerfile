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
# v14: giảm kích thước image 15.8 → 12.9 GB (bản lite: 11.1 GB), cải thiện layer cache
#      (v13 không phát hành, gộp vào v14)
#      - Layer order: COPY llama-server đặt cuối stage final. Đổi cờ build llama.cpp
#        không còn invalidate layer torch/requirements (~4.4 GB push/pull lại mỗi bản).
#      - apt: xóa *_static.a của cublas/cublasLt/curand/nvrtc/nvptxcompiler (~1.2 GB)
#        trong cùng RUN với apt install (xóa ở RUN sau không làm nhỏ image).
#        Giữ libcudart_static.a vì nvcc mặc định link cudart tĩnh.
#      - Base: cudnn-runtime → base (ARG CUDA_BASE). torch/vllm nạp cuBLAS/cuDNN/NCCL...
#        từ site-packages/nvidia qua RPATH, nên bản apt trùng lặp (~1.5 GB) không được dùng.
#        Thư viện còn thiếu lấp bằng LD_LIBRARY_PATH trỏ vào bản pip (đặt cuối để apt
#        được ưu tiên): libnccl cho llama-server, libnvJitLink/libcufile cho các .so trong venv.
#      - Bản lite (PRUNE_FLASHINFER_BLACKWELL_DC=1): xóa cubin sm100/103/110 của
#        flashinfer_cubin (~1.7 GB). Không dùng cho B100/B200/B300/GB200/Thor.
# =====================================================================

# =======================
# STAGE 1: BUILDER — chỉ compile llama-server (cần nvcc/cmake)
# =======================
# ARG toàn cục: phải đứng TRƯỚC FROM đầu tiên thì mới dùng được trong mọi lệnh FROM.
# base = chỉ có cudart + compat; các thư viện CUDA khác do apt (dependency của gói -dev)
# hoặc pip (wheel nvidia-*) cung cấp. Muốn quay lại base cũ (có sẵn cuDNN và CUDA runtime):
#   --build-arg CUDA_BASE=nvidia/cuda:13.0.3-cudnn-runtime-ubuntu24.04
ARG CUDA_BASE=nvidia/cuda:13.0.3-base-ubuntu24.04
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
FROM ${CUDA_BASE} AS final

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    VENV_PATH=/opt/venv \
    HF_HOME=/workspace/.cache/huggingface \
    # Include sm_120/12.0 cho Blackwell (RTX 5060 Ti) cùng các GPU phổ biến
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"

# Xóa thư viện tĩnh trong CÙNG RUN với apt install (xem ghi chú v14 mục 2).
# Giữ libcudart_static.a vì nvcc mặc định link cudart tĩnh.
# JIT/torch extension link động nên không cần các file .a bên dưới.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget python3 python3-venv python3-dev ca-certificates libgomp1 \
    cuda-nvcc-13-0 \
    cuda-nvrtc-dev-13-0 \
    libcurand-dev-13-0 \
    libcublas-dev-13-0 \
    && CUDA_LIB=/usr/local/cuda-13.0/targets/x86_64-linux/lib \
    && rm -f \
        "$CUDA_LIB/libcublasLt_static.a" \
        "$CUDA_LIB/libcublas_static.a" \
        "$CUDA_LIB/libcurand_static.a" \
        "$CUDA_LIB/libnvrtc_static.a" \
        "$CUDA_LIB/libnvptxcompiler_static.a" \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:${PATH}"
RUN pip install --upgrade pip

WORKDIR /workspace

# ---- Torch (cu130 index) ----
RUN pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu130

# ---- requirements.txt ----
COPY requirements.txt /workspace/requirements.txt

# Bản "lite": --build-arg PRUNE_FLASHINFER_BLACKWELL_DC=1 (xóa ~1.7 GB cubin sm100/103/110).
# Chỉ dùng cho GPU không phải Blackwell datacenter.
# Đặt ARG ngay trước RUN dùng nó để đổi giá trị không invalidate các layer phía trên.
ARG PRUNE_FLASHINFER_BLACKWELL_DC=0
RUN pip install -r /workspace/requirements.txt \
    && if [ "${PRUNE_FLASHINFER_BLACKWELL_DC}" = "1" ]; then \
         SP="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" \
         && find "$SP/flashinfer_cubin" -type f \( \
              -ipath '*sm100*' -o -ipath '*sm_100*' \
              -o -ipath '*sm103*' -o -ipath '*sm_103*' \
              -o -ipath '*sm110*' -o -ipath '*sm_110*' \
            \) -delete; \
       fi

# ---- llama-server binary từ builder ----
# Đặt cuối cùng: layer này đổi thường xuyên nhất, nên không được nằm trước các layer nặng.
# BUILD_SHARED_LIBS=OFF nên chỉ có 1 file; /opt/llama.cpp/bin trong LD_LIBRARY_PATH
# hiện không có tác dụng nhưng giữ lại phòng khi bật lại shared libs.
COPY --from=llama-builder /out/ /opt/llama.cpp/bin/
RUN ln -s /opt/llama.cpp/bin/llama-server /usr/local/bin/llama-server
# Base image không còn các thư viện CUDA runtime của apt (nccl, nvjitlink, cufile...)
# nên thêm thư mục thư viện của pip vào CUỐI LD_LIBRARY_PATH:
#   - nvidia/nccl/lib : libnccl.so.2 cho llama-server (nccl 2.28.9)
#   - nvidia/cu13/lib : libnvJitLink.so.13, libcufile.so.0... (check_image báo "not found"
#                       sau khi đổi base; ở v12 chúng được lấy từ thư viện apt của base cũ)
# Đặt ở CUỐI để thư viện của apt (nếu có) vẫn được ưu tiên, pip chỉ lấp chỗ thiếu.
# Torch dùng RPATH tới chính các thư mục này nên không bị ảnh hưởng.
# Python 3.12 là bản mặc định của Ubuntu 24.04; đổi đường dẫn nếu đổi phiên bản Python.
ENV LD_LIBRARY_PATH="/opt/llama.cpp/bin:${LD_LIBRARY_PATH}:${VENV_PATH}/lib/python3.12/site-packages/nvidia/nccl/lib:${VENV_PATH}/lib/python3.12/site-packages/nvidia/cu13/lib"

WORKDIR /workspace
CMD ["/bin/bash"]