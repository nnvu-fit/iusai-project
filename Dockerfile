FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 as base
ENV TZ=Asia/Ho_Chi_Minh \
    DEBIAN_FRONTEND=noninteractive

ARG PYTHON_VERSION=3.11.6
ARG JUPYTER_PASSWORD='password@1'

# Install Python and other dependencies
RUN apt update && apt install -y build-essential zlib1g-dev \
    libncurses5-dev libgdbm-dev libnss3-dev libsqlite3-dev \
    libssl-dev libreadline-dev libffi-dev libbz2-dev \
    liblzma-dev unzip wget curl

# Install python3.11.* from source code
RUN apt install -y python3.11

# Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.11 get-pip.py

# Install OpenCV
RUN apt install -y libopencv-dev python3-opencv

FROM base AS dev
# Install other python packages
RUN pip3 install numpy scipy matplotlib pandas scikit-learn
# Install PyTorch with cuda support (https://pytorch.org/get-started/locally/)
RUN pip3 install torch torchvision torchaudio
# Install other python packages
RUN pip3 install opencv-python opencv-contrib-python
# Install jupyterlab
RUN pip3 install jupyterlab notebook

FROM dev AS prod
WORKDIR /usr/app
EXPOSE 8888

CMD [ "jupyter", "lab", "--notebook-dir='/usr/app'", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token='password@1'" ]
