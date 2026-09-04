FROM nvcr.io/nvidia/pytorch:26.05-py3

# Runtime libraries for headless EGL/OpenGL rendering and MP4 encoding.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       ffmpeg \
       libegl1 \
       libgl1 \
       libglfw3 \
       libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
    mujoco==3.3.7 \
    mediapy==1.2.4 \
    fastapi==0.116.1 \
    "uvicorn[standard]==0.35.0"

ENV MUJOCO_GL=egl
WORKDIR /workspace

EXPOSE 8000
