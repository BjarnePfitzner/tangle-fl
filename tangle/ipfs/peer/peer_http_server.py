import logging

import prometheus_client
import requests
from flask import Flask, jsonify, Response

from .metrics.counter_metrics import counters_dic
from .metrics.histogram_metrics import histograms_dic

ADDRESS = '0.0.0.0'
PORT = 8000
IPFS_METRICS_ADDRESS = "http://localhost:5001/debug/metrics/prometheus"
app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


class PeerHttpServer(object):
    def __init__(self, tangle, peer):
        self.tangle = tangle
        self.peer = peer
        self.app = app
        self.logger = self.app.logger
        self.app.add_url_rule(rule='/api/transactions',
                              endpoint='generate_transactions_json',
                              view_func=self.generate_transactions_json)
        self.app.add_url_rule(rule='/api/peer',
                              endpoint='get_peer',
                              view_func=self.get_peer)
        self.app.add_url_rule(rule='/metrics',
                              endpoint='metrics',
                              view_func=self.metrics)

    def generate_transactions_json(self):
        n = [{
                'name': t.id,
                'time': t.metadata['time'],
                'timeCreated': t.metadata['timeCreated'] if 'timeCreated' in t.metadata else None,
                'peer': t.metadata['peer'],
                'parents': list(t.parents)
            } for _, t in self.tangle.transactions.items()]

        dic = {'nodes': n, 'genesis': str(self.tangle.genesis)}

        return jsonify(dic)

    def get_peer(self):
        return jsonify(self.peer)

    def metrics(self):
        res = []
        ipfs_metrics = requests.get(IPFS_METRICS_ADDRESS).text.split('\n')
        ipfs_metrics = [item + '\n' for item in ipfs_metrics]
        graphs = counters_dic.copy()
        graphs.update(histograms_dic)
        for k, v in graphs.items():
            res.append(prometheus_client.generate_latest(v))
        res.extend(ipfs_metrics)
        return Response(res, mimetype="text/plain")

    @staticmethod
    def start():
        app.run(host=ADDRESS, debug=False, use_reloader=False, port=PORT)
