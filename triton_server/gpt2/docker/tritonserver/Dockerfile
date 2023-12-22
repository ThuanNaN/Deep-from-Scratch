FROM nvcr.io/nvidia/tritonserver:23.07-py3

COPY requirements.txt .
COPY student_distill.pt .

RUN pip install --upgrade wheel setuptools pip
RUN pip install -r requirements.txt

WORKDIR /opt/tritonserver/
