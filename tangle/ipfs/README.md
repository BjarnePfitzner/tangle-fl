# IPFS Learning Tangle

## Overview

This repository contains everything required to run decentralized machine learning experiments in a containerized lab on one large machine or on a cluster of machines. The included prototype facilitates a Tangle architecture for distributed machine learning and IPFS for peer communication. The final version of this project can be run on arm32, arm64 and x64 node clusters in arbitrary networking situations with all-dockerized components and Kubernetes for orchestration and networking. Results of experiments (i.e., benchmarks and statistics) are gathered with Prometheus and visualized with Grafana. Furthermore, the tangle itself can be visualized with a custom-made visualization component.

## Key Components

### Peer

The peer service represents one participant of the decentralized machine learning network.

The bootstrap peer's IPFS ID is static (hardcoded in boostrap/config) i order to bootstrap the IPFS network. A new genesis transaction for the Tangle can be created by passing the `--create-genesis` parameter.

On startup, a peer loads its individual dataset (derived from its unique container IP address) and subscribes to the IPFS PubSub topic that is identified by the genesis transaction hash.

The tangle code (in [peer/tangle](peer/tangle)) is mostly copied from [paper experiments](https://github.com/osmhpi/learning-tangle).

The Listener class dispatches incoming events:

- Training is scheduled and performed periodically. The interval can be set by setting the `TRAINING_INTERVAL` and `ACTIVE_QUOTA` variables (see table below). The number of peers training each second on average is `1/TRAINING_INTERVAL*ACTIVE_QUOTA`.
- Whenever a new message (transaction) arrives through the PubSub topic, it is inserted into the local tangle object. If necessary, missing transactions are fetched from the IPFS file API by the TangleBuilder.

### Visualization

You can see how the tangle is being built in near real time using the visualization. To do that, use the LoadBalencer IP with port `9000` in your browser.

### Monitoring with Promethues & Grafana
You can access Grafana to see the various monitoring dashboards provided, using the LoadBalancer IP and with the port `3000`. The username and password for the Grafana panel is `admin`. You can have a closer look at the scrape jobs and the metrics in the Prometheus dashboard. To reach that use the LoadBalancer IP and the port `9090`.

## Getting Started

### Prerequisites

- Docker installed
- Docker-Compose installed
- Kubernetes installed and cluster configured

If there are further requirements you come across that we did not explicitly name here, please let us know.

### Build and Publish

 1. Login to DockHub with `docker login`. The username is hpimpss2020 and the password can be obtained from the creators of this project.
 2. Download the training data and genesis transaction from [here](https://owncloud.hpi.de/s/QApb0yznb6Its8a)
 3. Put the data folder (contains training data) into the peer folder
 4. Put the `genesis.npy` (example genesis transaction) file from the tangle folder into the data folder
 5. The last step depends on the target architecture for the network:

For x86 peer clusters, run

```bash
docker-compose build && docker-compose push
```

For arm32 peer clusters, run

```bash
docker-compose -f docker-compose.yml -f docker-compose.arm.yml build && docker-compose -f docker-compose.yml -f docker-compose.arm.yml push
```

For arm64 peer clusters, run

```bash
docker-compose -f docker-compose.yml -f docker-compose.arm.yml -f docker-compose.arm64.yml build && docker-compose -f docker-compose.yml -f docker-compose.arm.yml -f docker-compose.arm64.yml push
```

### Running with Kubernetes

From the [deploy file](https://gitlab.hpi.de/osm/mpss2020/ipfs-learning-tangle/-/blob/master/k8s/peer-deployment.yaml), you can pass the following environment variables to configure the peers.

| ENV Name |      Description      | Values |
|----------|:-------------------------:|:------------------:|
| STORAGE | Type of data storage used to fetch Tangle data (e.g., weights) | ipfs
| MESSAGE_BROKER | Message passing used to transmit Tangle messages (e.g., new transaction announcments) | ipfs
| MODEL | Machine learning (or fake ML) module used for training | no_tf  or femnist
| TIMEOUT | Timeout for connections to the IPFS daemon | none or positive integer
| TRAINING_INTERVAL | Training interval period | positive integer
| NUM_OF_TIPS | Number of preceding tips approved by a Tangle transaction | positive integer
| NUM_OF_SAMPLING_ROUND | Number of samplings performed each learning round | opsitive integer
| ACTIVE_QUOTA | Probability of one peer becoming active in a training period | Floating point number between 0 and 1
| CONNECTION_PRUNING_LOW | Minimum number of open IPFS peer connections (see [IPFS](https://github.com/ipfs/go-ipfs/blob/master/docs/config.md#swarmconnmgrlowwater)) | positive integer
| CONNECTION_PRUNING_HIGH | Maximum number of open IPFS peer connections (see [IPFS](https://github.com/ipfs/go-ipfs/blob/master/docs/config.md#swarmconnmgrhighwater)) | positive integer
| CONNECTION_PRUNING_GRACE_PERIOD | Time in seconds after which open IPFS peer connections become relevant for pruning (see [IPFS](https://github.com/ipfs/go-ipfs/blob/master/docs/config.md#swarmconnmgrgraceperiod)) | positive integer
| LOGGER | Log output destination | print or file
| LOGGING_LEVEL | Logging verbosity | info or warn or error
| IPFS_DATASTORE_STORAGEMAX | Limit for the storage usage of each IPFS peer (see [IPFS](https://github.com/ipfs/go-ipfs/blob/master/docs/config.md#datastorestoragemax)) | e.g. 10GB
| IPFS_DATASTORE_GCPERIOD | How often IPFS garbage collection is run (see [IPFS](https://github.com/ipfs/go-ipfs/blob/master/docs/config.md#datastorestoregcperiod)) | e.g. 1h
| IPFS_REPROVIDER_INTERVAL | How often provided blocks are reannounced to the IPFS network (see [IPFS](https://github.com/ipfs/go-ipfs/blob/master/docs/config.md)) | e.g. 1h

Running the experiment prototype on a Kubernetes cluster requires execution of the following commands:

```bash
kubectl create secret generic regcred \
    --from-file=.dockerconfigjson=<path/to/.docker/config.json> \
    --type=kubernetes.io/dockerconfigjson
```

```bash
kubectl apply -f k8s/
kubectl scale --replicas=3 -f k8s/peer-deployment.yaml
```

If you want to deploy on a specific architecture (such as arm64) change the image-names in k8s/*-deployment.yaml before starting.
