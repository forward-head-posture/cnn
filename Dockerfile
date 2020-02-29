FROM tensorflow/tensorflow:2.0.0-gpu-py3
WORKDIR /app

COPY . .
RUN pip install fire
RUN pip install -e ./

ENTRYPOINT [ "cnn" ]
