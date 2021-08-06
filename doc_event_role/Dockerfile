# FROM nvcr.io/nvidia/pytorch:18.12-py3
# FROM nvcr.io/nvidia/pytorch:19.11-py3
FROM nvcr.io/nvidia/pytorch:20.03-py3

RUN mkdir -p /der
WORKDIR /der

COPY . /der

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN pip install jupyter
# RUN pip install -U pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

CMD ["/bin/bash"]
