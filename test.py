import os
import time
import tensorflow as tf
import numpy as np
import csv
from configs import Config
from nets.model import RNN_cell
from data.testset import TestReader

"""main function"""
def main(FLAGS):
    # Define data reader.
    testset = TestReader(FLAGS.DATA_DIR, FLAGS.SAMPLING_MINUTE, FLAGS.SAMPLING_SIZE, FLAGS.BATCH_SIZE)

    # Define model.
    model = RNN_cell(FLAGS.INPUT_SIZE, FLAGS.HIDDEN_LAYER_SIZE, FLAGS.TARGET_SIZE)

    # Define label and model output.
    outputs = model.get_outputs()
    last_output = outputs[-1]
    output = tf.nn.softmax(last_output)
    output_class = tf.argmax(output, 1)

    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.MODEL_PATH))
        test_x = testset.X_test
        test_predict = sess.run(output_class, feed_dict={model._inputs: test_x})

        f = open('output.csv', 'w', encoding='utf-8')
        wr = csv.writer(f)
        for date in np.arange(365):
            row_list = test_predict[date*8:(date+1)*8]
            wr.writerow(row_list)
        f.close()


    print("Finished!")


if __name__ == '__main__':
    FLAGS = Config()
    FLAGS.MODE = 'test'
    main(FLAGS)

