#https://schultz-christian.medium.com/how-to-run-tensorflow-in-docker-on-an-apple-silicon-mac-c56f3127b696


FROM --platform=linux/aarch64 python:3.10 
ARG TENSORFLOW_VERSION=v2.9.1 
ARG DEBIAN_FRONTEND=noninteractive 
RUN apt update && apt install -y build-essential python3-dev pkg-config zip zlib1g-dev unzip curl wget git htop openjdk-11-jdk liblapack3 libblas3 libhdf5-dev npm 
RUN npm install -g @bazel/bazelisk 
RUN pip install six numpy grpcio h5py packaging opt_einsum wheel requests 
RUN git clone https://github.com/tensorflow/tensorflow.git && mkdir -p /wheels/tensorflow
WORKDIR tensorflow 
RUN git checkout $TENSORFLOW_VERSION 
RUN bazel build -c opt --verbose_failures //tensorflow/tools/pip_package:build_pip_package
RUN /bazel-bin/tensorflow/tools/pip_package/build_pip_package /wheels/tensorflow
CMD ["pip", "install", "--extra-index-url", "/wheels", "tensorflow"]

RUN pip install -r requirements.txt --extra-index-url=http://mydomain.com --trusted-host=mydomain.com
