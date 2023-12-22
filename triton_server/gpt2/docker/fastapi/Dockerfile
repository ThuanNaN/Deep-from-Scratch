FROM nvcr.io/nvidia/tritonserver:23.07-py3-sdk

COPY requirements.txt .

RUN pip install --upgrade wheel setuptools pip
RUN pip install -r requirements.txt

WORKDIR /workspace/
