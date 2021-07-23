FROM ipfs/go-ipfs:v0.9.0

COPY entrypoint /entrypoint
COPY config /config

EXPOSE 4001

RUN chmod +x /entrypoint
ENTRYPOINT [ "/sbin/tini", "--", "/entrypoint" ]
CMD [ "daemon", "--migrate=true", "--enable-pubsub-experiment", "--enable-gc" ]
