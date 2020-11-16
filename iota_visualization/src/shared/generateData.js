export const loadTangle = async ({peerNumber}) => {
    let address = await loadPeerAddress({peerNumber});

    let nodes = []
    let links = []
    if (address == null) {
        return {nodes, links};
    }

    const resp = await fetch(`/api/transactions/${address}`, {
        method: 'get',
        headers: {'Content-Type': 'application/json'},
    }).catch(exp => {
        console.log("error in transactions: ", exp);
        return {};
    });

    const data = await resp.json();

    nodes = data.nodes.map(x => ({...x, x: 300, y: 200,}));
    links = data.nodes.flatMap(x => x.parents.map(p => ({
        source: nodes.find(n => n.name === x.name),
        target: nodes.find(n => n.name === p)
    })));

    return {nodes, links};
}

export const loadPeer = async ({peerNumber}) => {
    let address = await loadPeerAddress({peerNumber});

    const resp = await fetch(`/api/peer/${address}`, {
        method: 'get',
        headers: {'Content-Type': 'application/json'},
    }).catch(exp => {
        console.log("error in peer: ", exp);
        return {};
    });

    if (address == null) {
        return {client_id: "-"}
    }

    const peer = await resp.json();
    console.log("peer is: ", peer)
    return {client_id: peer.client_id}
}

export const numberOfPeers = async () => {
    const addressList = await getAddressList()
    console.log("adderss list is: ", addressList)
    console.log("number of peers: ", addressList.length)
    return addressList.length
}

const loadPeerAddress = async ({peerNumber}) => {
    const addressList = await getAddressList()

    if (parseInt(peerNumber) > addressList.length) {
        return null;
    }

    console.log("address is: ", addressList[parseInt(peerNumber) - 1])

    return addressList[parseInt(peerNumber) - 1]
}

const getAddressList = async () => {
    const res = await fetch(`/api/address`, {
        method: 'get',
        headers: {'Content-Type': 'application/json'},
    }).catch(exp => {
        console.log("error in address: ", exp);
        return {};
    });
    const addresses = await res.json();
    return addresses;
}