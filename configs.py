from pathlib import Path
import numpy as np
import tensorflow as tf
############################################# Experiment Configuration #############################################
"""
* Experiment Details

"""
###################################################################################################################
class Config(object):
    """ Directory settings """
    MODEL_PATH = Path('model')                          # Path to save model weight
    MODEL_PATH.mkdir(exist_ok=True)
    LOG_PATH = Path('logs')                             # Path to save log summary
    LOG_PATH.mkdir(exist_ok=True)
    DATA_DIR = Path('data/dataset')
    DATA_DIR.mkdir(exist_ok=True)
    TEST_DATA_DIR = Path('data/dataset')
    DATA_DIR.mkdir(exist_ok=True)

    """ Training settings """
    MODE = 'train'                                  # 'train' or 'test'
    EPOCH = 100                                         # training epoch
    LEARNING_RATE = 1e-2                                # learning rate
    LR_DECAY_EPOCH = 10
    HIDDEN_LAYER_SIZE = 96
    DISPLAY_STEP = 40
    SAVE_STEP = 1000
    SAVE_SUMMARY = False
    USE_JSON = True

    """ data settings """
    BATCH_SIZE = 100
    INPUT_SIZE = 7
    SAMPLING_MINUTE = 30
    SAMPLING_SIZE = 50
    TARGET_SIZE = 10

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

    def tf_Weighted_RMSE(self, labels, predict):
        label_one_hot = tf.one_hot(tf.cast(labels, tf.int32), depth=self.TARGET_SIZE)
        weights = tf.stack([labels / tf.reduce_sum(labels)]*self.TARGET_SIZE, 1)
        eval = tf.sqrt(tf.reduce_sum(tf.square(predict-label_one_hot)*weights))
        return eval

    def Weighted_RMSE(self, labels, predict):
        labels = np.array(labels)
        predict = np.array(predict)
        weights = labels / np.sum(labels)
        eval = np.sqrt(np.sum(np.square(predict-labels) * weights))
        return eval


    ######################################

