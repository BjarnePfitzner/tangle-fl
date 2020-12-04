from prometheus_client import Counter

counters_dic = {
    'transactions_received_total': Counter('peer_transactions_received_total',
                                           'The total number of transactions received'),
    'transactions_published_total': Counter('peer_transactions_published_total',
                                            'The total number of transactions published'),
    'transactions_published_error_total': Counter('peer_transactions_publish_error_total',
                                                  'The total number of transactions publish error'),
    'subscribers_total': Counter('peer_subscribers_total', 'The total number of subscribers'),
    'subscribers_exit_total': Counter('peer_subscribers_exit_total',
                                      'The total number of subscribers exit'),
    'trainings_total': Counter('peer_trainings_total', 'The total number of trainings'),
    'training_errors_total': Counter('peer_training_error_total', 'The total number of training error'),
    'publish_errors_total': Counter('peer_publish_error_total', 'The total number of publish error'),
}


def increment_count_transaction_received():
    counters_dic['transactions_received_total'].inc()


def increment_count_transaction_published():
    counters_dic['transactions_published_total'].inc()


def increment_count_transaction_publish_error():
    counters_dic['transactions_published_error_total'].inc()


def increment_counter_subscriber():
    counters_dic['subscribers_total'].inc()


def increment_counter_subscriber_exit():
    counters_dic['subscribers_exit_total'].inc()


def increment_count_training():
    counters_dic['trainings_total'].inc()


def increment_count_training_error():
    counters_dic['training_errors_total'].inc()


def increment_count_publish_error():
    counters_dic['publish_errors_total'].inc()
