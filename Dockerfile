FROM tensorflow/tensorflow:2.2.0rc2-gpu-py3
WORKDIR /app

RUN pip install fire
COPY . .
RUN python setup.py install

ENTRYPOINT [ "cnn" ]
