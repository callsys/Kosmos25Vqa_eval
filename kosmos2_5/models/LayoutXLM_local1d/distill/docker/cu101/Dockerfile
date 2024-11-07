FROM mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04
RUN pip install --no-cache-dir pip==20.1.1 && \
    pip install --no-cache-dir torch==1.8.1 torchvision==0.9.1 -f https://download.pytorch.org/whl/cu101/torch_stable.html

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         ca-certificates \
         curl \
         htop \
         vim \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

RUN set -ex \
    && wget -O azcopy.tar.gz https://aka.ms/downloadazcopy-v10-linux \
    && tar -xzf azcopy.tar.gz && rm -f azcopy.tar.gz \
    && cp ./azcopy_linux_amd64_*/azcopy /usr/local/bin/. \
    && chmod +x /usr/local/bin/azcopy \
    && rm -rf azcopy_linux_amd64_*