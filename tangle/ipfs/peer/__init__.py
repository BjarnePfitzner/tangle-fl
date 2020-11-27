import logging.config
from os import path


log_file_path = path.join(path.dirname(path.abspath(__file__)), 'log.config')
logging.config.fileConfig(log_file_path)
logger = logging.getLogger('ipfs_learning_tangle')
