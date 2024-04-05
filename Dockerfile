FROM nvidia/cuda:12.3.2-devel-ubuntu22.04
RUN apt update && apt install -y python3 python3-pip
ADD requirements.txt /
RUN pip install --no-cache -r /requirements.txt
