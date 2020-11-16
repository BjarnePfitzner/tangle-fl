FROM python:3.7
ARG IPFS_VERSION=0.6.0

WORKDIR /
RUN mkdir logs

RUN apt-get update \
    && apt-get install -y dnsutils

RUN mkdir /tmp/go-ipfs \
    && curl -sL https://ipfs.io/ipns/dist.ipfs.io/go-ipfs/v${IPFS_VERSION}/go-ipfs_v${IPFS_VERSION}_linux-amd64.tar.gz | tar xz -C /tmp/go-ipfs --strip 1 \
    && cd /tmp/go-ipfs && ./install.sh \
    && rm -rf /tmp/ipfs

WORKDIR /app

COPY tangle/ipfs/peer/docker/requirements.txt .
RUN pip3 install -r requirements.txt

COPY tangle/ipfs/peer/docker/prepare_data.sh /usr/bin/prepare_data.sh
RUN /usr/bin/prepare_data.sh

COPY tangle/ipfs/peer/docker/ipfs_entrypoint /usr/bin/ipfs_entrypoint
COPY tangle /usr/local/lib/python3.7/site-packages/tangle/
COPY tangle/ipfs/peer/docker/entrypoint.sh .
COPY tangle/ipfs/peer/docker/manual_garbage_collection.sh .

EXPOSE 4001
EXPOSE 8000

ENV PYTHONUNBUFFERED=TRUE
RUN chmod +x ./entrypoint.sh
RUN chmod +x ./manual_garbage_collection.sh
ENTRYPOINT ["./entrypoint.sh"]
