FROM python:3.7


WORKDIR /
RUN mkdir logs


RUN apt-get update \
    && apt-get install -y dnsutils

RUN mkdir /tmp/go-ipfs \
    && curl -sL https://ipfs.io/ipns/dist.ipfs.io/go-ipfs/v0.6.0/go-ipfs_v0.6.0_linux-amd64.tar.gz | tar xz -C /tmp/go-ipfs --strip 1 \
    && cd /tmp/go-ipfs && ./install.sh \
    && rm -rf /tmp/ipfs

WORKDIR /app
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY ipfs_entrypoint /usr/bin/ipfs_entrypoint
COPY . /usr/local/lib/python3.7/site-packages/learner/
COPY ./entrypoint.sh .
COPY ./manual_garbage_collection.sh .
COPY ./prepare_data.sh /usr/bin/prepare_data.sh
RUN /usr/bin/prepare_data.sh


EXPOSE 4001
EXPOSE 8000

ENV PYTHONUNBUFFERED=TRUE
RUN chmod +x ./entrypoint.sh
RUN chmod +x ./manual_garbage_collection.sh
ENTRYPOINT ["./entrypoint.sh"]
