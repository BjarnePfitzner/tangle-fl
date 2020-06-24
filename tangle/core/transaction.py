import numpy as np

class Transaction:
    def __init__(self, parents, client_id, cluster_id, id=None, tag=None, malicious=False):
        self.parents = parents
        self.client_id = client_id
        self.cluster_id = cluster_id
        self.tag = tag
        self.id = id
        self.malicious = malicious
        self.tangle_dir = tangle_dir

    def height(self, tangle):
      pass

    def name(self):
        return self.id

    def add_tag(self, tag):
        self.tag = tag
