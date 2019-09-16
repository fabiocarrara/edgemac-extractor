FROM bvlc/caffe:gpu

RUN pip install --no-cache-dir --upgrade scikit-image && \
    pip install --no-cache-dir \
        flask-restful \
        opencv-python \
        opencv-contrib-python \
        sklearn \
        tornado \
        tqdm

RUN mkdir -p /code
WORKDIR /code
ADD . /code

RUN cd models && ./download_models.sh && cd ..

