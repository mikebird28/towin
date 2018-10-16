#Base Image
FROM ubuntu:latest
MAINTAINER mikebird28

WORKDIR /app
ADD . /app

#update repsitories
RUN apt update && \
    apt install -y curl cmake  gcc g++ git systemd python3-dev  && \
    rm -rf /var/lib/apt/lists/*

#install loging agent
#install pip
RUN curl -kL https://bootstrap.pypa.io/get-pip.py | python3

#install dependent libraries
RUN pip install --upgrade pip &&\
    pip install --upgrade setuptools &&\
    pip install -r requirement.txt

#install lightgbm
RUN git clone --recursive https://github.com/Microsoft/LightGBM && \
    cd LightGBM/python-package && \
    python3 setup.py install && \
    rm -rf LightGBM

#RUN script
CMD ["python3","-u","policy.py"]

