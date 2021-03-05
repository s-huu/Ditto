SIM_TIMES = ['small', 'medium', 'large']

MAIN_PARAMS = {  # (tot_num_rounds, eval_every_num_rounds, clients_per_round)
    'so': {
        'small': (10, 2, 2),
        'medium': (16, 2, 2),
        'large': (1000, 50, 100)
    }
}


MODEL_PARAMS = {
    # Stackoverflow
    'so.erm_cnn_log_reg': (0.01, 10000, 500, 100)
}

MAX_UPDATE_NORM = 100000  # reject all updates larger than this amount

ACCURACY_KEY = 'accuracy'
BYTES_WRITTEN_KEY = 'bytes_written'
BYTES_READ_KEY = 'bytes_read'
LOCAL_COMPUTATIONS_KEY = 'local_computations'
NUM_ROUND_KEY = 'round_number'
NUM_SAMPLES_KEY = 'num_samples'
CLIENT_ID_KEY = 'client_id'
AVG_LOSS_KEY = 'avg_loss'

# List of regularization parameters tested for validation
REGULARIZATION_PARAMS = [10**i for i in range(-10, -4)]

class OptimLoggingKeys:
    TRAIN_ACCURACY_KEY = 'train_accuracy'
    TRAIN_LOSS_KEY = 'train_loss'
    EVAL_ACCURACY_KEY = 'test_accuracy'
    EVAL_LOSS_KEY = 'test_loss'
    DIFF_NORM = 'norm_of_difference'

TRAINING_KEYS = {OptimLoggingKeys.TRAIN_ACCURACY_KEY,
                 OptimLoggingKeys.TRAIN_LOSS_KEY,
                 OptimLoggingKeys.EVAL_LOSS_KEY}

AGGR_MEAN = 'mean'
AGGR_MEDIAN = 'median'
AGGR_KRUM = 'krum'
AGGR_MULTIKRUM = 'multikrum'