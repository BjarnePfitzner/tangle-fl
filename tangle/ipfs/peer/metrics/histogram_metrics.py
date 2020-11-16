from prometheus_client import Histogram

histograms_dic = {
    "time_training": Histogram('peer_training_time_seconds', 'Histogram for the duration in seconds.'),
    "time_between_training": Histogram('peer_time_between_training_seconds',
                                       'Histogram for the duration in seconds.'),
    "consensus_accuracy": Histogram('peer_consensus_accuracy_percent', 'Histogram for the consensus'),
    "consensus_loss": Histogram('peer_consensus_model_loss_percent', 'Histogram for the consensus model loss'),
    "model_accuracy": Histogram('peer_model_accuracy_percent', 'Histogram for the model accuracy'),
    "model_loss": Histogram('peer_model_loss_percent', 'Histogram for the model loss'),
}


def observe_time_training(time: float):
    histograms_dic['time_training'].observe(time)


def observe_time_between_training(time: float):
    histograms_dic['time_between_training'].observe(time)


def observe_consensus_accuracy(accuracy: float):
    histograms_dic['consensus_accuracy'].observe(accuracy)


def observe_consensus_model_loss(loss: float):
    histograms_dic['consensus_loss'].observe(loss)


def observe_model_accuracy(accuracy: float):
    histograms_dic['model_accuracy'].observe(accuracy)


def observe_model_loss(loss: float):
    histograms_dic['model_loss'].observe(loss)
