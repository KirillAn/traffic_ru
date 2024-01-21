FROM nvcr.io/nvidia/tritonserver:23.02-py3
#FROM "nvcr.io/nvidia/tritonserver:22.06-py3"

COPY models /models


# Install dependencies
RUN pip install opencv-python && \
    apt update 

CMD ["tritonserver", "--model-repository=/models" ]
