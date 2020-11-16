from .storage import Storage


class IPFSStorage(Storage):

    def __init__(self, ipfs_client):
        self._ipfs_client = ipfs_client

    # Are used to store & retrieve weights
    # Returns the file path or Hash
    def add_file(self, content):
        try:
            return self._ipfs_client.add(content)['Hash']
        except:
            return None

    # Returns the file content or Hash
    def get_file(self, path: str):
        try:
            return self._ipfs_client.cat(path)
        except:
            return None

    # Are used to store transactions and to retrieve missed transactions
    # Returns the file path or Hash
    def add_json(self, value: dict):
        try:
            return self._ipfs_client.add_json(value)
        except:
            return None

    def get_json(self, path: str):
        try:
            return self._ipfs_client.get_json(path)
        except:
            return None
