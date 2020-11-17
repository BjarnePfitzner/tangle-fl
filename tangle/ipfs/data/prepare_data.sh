curl https://owncloud.hpi.de/s/QApb0yznb6Its8a/download --output /tmp/data.zip
unzip /tmp/data.zip -d /tmp
cp -r /tmp/ipfs-learning-tangle-data/data /data
cp /tmp/ipfs-learning-tangle-data/tangle/genesis.npy /data/
