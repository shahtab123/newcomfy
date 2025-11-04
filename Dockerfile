# Build argument for base image selection
# NOTE: Using devel (not runtime) because we build CUDA extensions (SageAttention, WanVideoWrapper) during image build
ARG BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

# Stage 1: Base image with common dependencies
FROM ${BASE_IMAGE} AS base

# Build arguments for this stage with sensible defaults for standalone builds
ARG COMFYUI_VERSION=latest
ARG CUDA_VERSION_FOR_COMFY
ARG ENABLE_PYTORCH_UPGRADE=false
ARG PYTORCH_INDEX_URL

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1
# Speed up some cmake builds
ENV CMAKE_BUILD_PARALLEL_LEVEL=8
# Prevent pip from asking for confirmation during uninstall steps in custom nodes
ENV PIP_NO_INPUT=1
# Increase timeout for long-running operations
ENV UV_HTTP_TIMEOUT=300

# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    gcc \
    g++ \
    build-essential \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Add a fake libcuda.so symlink for serverless environments (no GPU driver at build time)
RUN mkdir -p /usr/lib/x86_64-linux-gnu && \
    ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so || true

# Install uv (latest) using official installer and create isolated venv
RUN wget -qO- https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv \
    && ln -s /root/.local/bin/uvx /usr/local/bin/uvx \
    && uv venv /opt/venv

# Use the virtual environment for all subsequent commands
ENV PATH="/opt/venv/bin:${PATH}"

# Set correct CUDA environment variables for Triton and PyTorch
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/compat:${LD_LIBRARY_PATH}"

# Install comfy-cli + dependencies needed by it to install ComfyUI
RUN uv pip install comfy-cli pip setuptools wheel

# Install ComfyUI
RUN if [ -n "${CUDA_VERSION_FOR_COMFY}" ]; then \
      /usr/bin/yes | comfy --workspace /comfyui install --version "${COMFYUI_VERSION}" --cuda-version "${CUDA_VERSION_FOR_COMFY}" --nvidia; \
    else \
      /usr/bin/yes | comfy --workspace /comfyui install --version "${COMFYUI_VERSION}" --nvidia; \
    fi

# Upgrade PyTorch if needed (for newer CUDA versions)
RUN if [ "$ENABLE_PYTORCH_UPGRADE" = "true" ]; then \
      uv pip install --force-reinstall torch torchvision torchaudio --index-url ${PYTORCH_INDEX_URL}; \
    fi

# Set target GPU architectures for SageAttention compilation (no GPU at build time)
# Targeting A100 (8.0), RTX 3090/A40 (8.6), RTX 4090/L40S/Ada (8.9)
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"

# Install SageAttention 2.2.0 for optimized attention performance
# Requirements: python>=3.9, torch>=2.3.0, triton>=3.0.0
# See: https://github.com/thu-ml/SageAttention
# Version auto-compatible with PyTorch installed by ComfyUI
RUN uv pip install --no-build-isolation sageattention==2.2.0

# Install dependencies for InfiniteTalk workflow (wav2vec2, HuggingFace models, etc.)
# Versions auto-resolved to be compatible with existing PyTorch installation
RUN uv pip install huggingface_hub einops transformers accelerate xformers

# Change working directory to ComfyUI
WORKDIR /comfyui

# Support for the network volume
ADD src/extra_model_paths.yaml ./

# Add script to install custom nodes (must be before using it)
COPY scripts/comfy-node-install.sh /usr/local/bin/comfy-node-install
RUN chmod +x /usr/local/bin/comfy-node-install

# Install custom nodes
RUN comfy-node-install \
    https://github.com/kijai/ComfyUI-WanVideoWrapper \
    https://github.com/rgthree/rgthree-comfy \
    https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite

# Prebuild WanVideoWrapper CUDA extensions to avoid runtime compilation (if it exists)
RUN if [ -d /comfyui/custom_nodes/ComfyUI-WanVideoWrapper ]; then \
        echo "Prebuilding WanVideoWrapper CUDA extensions..."; \
        cd /comfyui/custom_nodes/ComfyUI-WanVideoWrapper && \
        uv pip install -e . && \
        echo "WanVideoWrapper CUDA extensions prebuilt successfully"; \
    else \
        echo "WARNING: ComfyUI-WanVideoWrapper not found - extensions will compile at runtime (slower first run)"; \
    fi

# Go back to the root
WORKDIR /

# Install Python runtime dependencies for the handler
RUN uv pip install runpod requests websocket-client

# Add application code and scripts
ADD src/start.sh handler.py test_input.json ./
RUN chmod +x /start.sh

# Copy helper script to switch Manager network mode at container start
COPY scripts/comfy-manager-set-mode.sh /usr/local/bin/comfy-manager-set-mode
RUN chmod +x /usr/local/bin/comfy-manager-set-mode

# Set the default command to run when starting the container
CMD ["/start.sh"]

# Stage 2: Download models
FROM base AS downloader

ARG HUGGINGFACE_ACCESS_TOKEN
# Set default model type if none is provided
ARG MODEL_TYPE=flux1-dev-fp8

# Change working directory to ComfyUI
WORKDIR /comfyui

# Create necessary directories upfront
RUN mkdir -p models/checkpoints models/vae models/unet models/clip

# Download checkpoints/vae/unet/clip models to include in image based on model type

RUN if [ "$MODEL_TYPE" = "sd3" ]; then \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/checkpoints/sd3_medium_incl_clips_t5xxlfp8.safetensors https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/sd3_medium_incl_clips_t5xxlfp8.safetensors; \
    fi



RUN if [ "$MODEL_TYPE" = "flux1-dev-fp8" ]; then \
      wget -q -O models/checkpoints/flux1-dev-fp8.safetensors https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors; \
    fi

# Stage 3: Final image
FROM base AS final

# Copy models from stage 2 to the final image
COPY --from=downloader /comfyui/models /comfyui/models