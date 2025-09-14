FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS base

ARG USERID=1000
ARG GID=1000
ARG UNAME=user
ARG PASSWORD=password

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update &&         apt-get install -y --no-install-recommends             build-essential             git             curl             ca-certificates             wget             rsync             python3-dev             python3-pip             libgl1-mesa-glx             libglib2.0-0             libgtk2.0-dev             libtbb-dev             pkg-config             ffmpeg             sudo         && rm -rf /var/lib/apt/lists/*

RUN if ! getent group ${GID} > /dev/null; then             groupadd -g ${GID} ${UNAME};         fi &&         useradd -m -s /bin/bash -u ${USERID} -g ${GID} -o ${UNAME} &&         echo "${UNAME}:${PASSWORD}" | chpasswd &&         usermod -aG sudo ${UNAME}

WORKDIR /workspace/drivestudio

FROM base AS builder

COPY requirements.txt .

RUN python3 -m pip install --upgrade pip &&         pip install -r requirements.txt &&         pip install -U openmim &&         mim install mmengine &&         mim install "mmcv==2.1.0" &&         pip install "mmsegmentation>=1.0.0" &&         pip install gsplat --index-url https://docs.gsplat.studio/whl/pt22cu121 &&         pip install git+https://github.com/facebookresearch/pytorch3d.git &&         pip install git+https://github.com/NVlabs/nvdiffrast &&         pip install git+https://github.com/tier4/t4-devkit.git@v0.1.0 &&         pip cache purge

FROM builder AS smplx

RUN apt-get update &&         apt-get install -y --no-install-recommends             libegl1-mesa-dev libgl1-mesa-glx libgles2-mesa-dev mesa-utils             libglapi-mesa libgl1-mesa-dri xvfb libglu1-mesa-dev             libosmesa6-dev libosmesa6         && rm -rf /var/lib/apt/lists/*

RUN pip install git+https://github.com/brjathu/PHALP.git

RUN pip uninstall -y PyOpenGL PyOpenGL_accelerate || true &&         pip install git+https://github.com/mmatl/pyopengl.git

COPY third_party/smplx ./third_party/smplx/
RUN cd third_party/smplx && pip install -e . && pip cache purge

COPY third_party/Humans4D ./third_party/Humans4D/
RUN cd third_party/Humans4D && pip install -e .[all]

ENV PYTHONPATH=/workspace/drivestudio:$PYTHONPATH
COPY docker/docker-entrypoint.smplx-only.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh &&         mkdir -p /workspace/weights /workspace/data &&         chown -R ${USERID}:${GID} /workspace

USER ${UNAME}
WORKDIR /workspace/drivestudio
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["bash"]
