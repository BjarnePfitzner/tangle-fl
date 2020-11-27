import logging
import subprocess
import threading

import requests
from flask import Flask, jsonify, abort
from flask_cors import CORS

ADDRESS = '0.0.0.0'
PORT = 9000
app = Flask(__name__, static_url_path='', static_folder='public')
CORS(app)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


@app.route('/')
def root():
    return app.send_static_file('index.html')


@app.route('/api/transactions/<peer>')
def get_transactions(peer):
    try:
        address = 'http://{}:8000/api/transactions'.format(peer)
        response = requests.get(address)
        return jsonify(response.json())
    except requests.exceptions.RequestException:
        response = {'code': 400, 'message': 'error in transactions'}
        return abort(400, jsonify(response))


@app.route('/api/peer/<peer>')
def get_id(peer):
    try:
        address = 'http://{}:8000/api/peer'.format(peer)
        response = requests.get(address)
        return jsonify(response.json())
    except requests.exceptions.RequestException:
        response = {'code': 400, 'message': 'error in peer'}
        return abort(400, jsonify(response))


@app.route('/api/address')
def get_peer_addresses():
    bash_command = ['nslookup', 'peer'] # k8s: peer-service
    output = subprocess.check_output(bash_command).decode('utf-8')
    lines = output.split('\n')
    addresses = [x.replace('Address: ', '').strip() for x in lines if x.startswith('Address:')][1:]
    return jsonify(addresses)


print('serving at port', PORT)
threading.Thread(app.run(host=ADDRESS, debug=False, use_reloader=False, port=PORT)).start()
