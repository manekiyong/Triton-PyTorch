FROM nvcr.io/nvidia/tritonserver:22.02-py3

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

CMD ["tritonserver", "--model-repository=/models"]
