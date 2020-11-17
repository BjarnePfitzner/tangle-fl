FROM alpine

RUN apk add --no-cache curl

COPY prepare_data.sh /usr/bin/prepare_data.sh
RUN prepare_data.sh
