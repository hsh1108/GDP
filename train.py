import os
import time
import tensorflow as tf
import numpy as np

from configs import Config
from nets.model import RNN_cell
from data.reader import DataReader

"""main function"""
def main(FLAGS):
    # Define data reader.
    dataset = DataReader(FLAGS.DATA_DIR, FLAGS.SAMPLING_MINUTE, FLAGS.SAMPLING_SIZE, FLAGS.BATCH_SIZE, FLAGS.USE_JSON)

    # Define model.
    model = RNN_cell(FLAGS.INPUT_SIZE, FLAGS.HIDDEN_LAYER_SIZE, FLAGS.TARGET_SIZE)

    # Define label and model output.
    label = tf.placeholder(tf.float32, shape=[None], name='labels')
    label_one_hot = tf.one_hot(tf.cast(label, tf.int32), depth=FLAGS.TARGET_SIZE)
    outputs = model.get_outputs()
    last_output = outputs[-1]
    output = tf.nn.softmax(last_output)
    output_class = tf.argmax(output, 1)

    # Define cross entropy loss.
    cross_entropy = -tf.reduce_mean(label_one_hot* tf.log(output))
    #cross_entropy = FLAGS.tf_Weighted_RMSE(label, output)
    #cross_entropy = tf.reduce_mean(tf.square(output - label))
    #cross_entropy = tf.reduce_mean(label*output)

    # Define adam training optimizer.
    learning_rate = tf.placeholder(tf.float32)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    saved_eval= 10.0

    # Define log summary
    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('loss', cross_entropy)

    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.initialize_all_variables())
        merged = tf.summary.merge_all()
        if FLAGS.SAVE_SUMMARY:
            writer = tf.summary.FileWriter(FLAGS.LOG_PATH, sess.graph)
        saver = tf.train.Saver(max_to_keep=10)
        for epoch in range(0, FLAGS.EPOCH):
            start_time = time.time()
            for num in range(0, dataset.train_batch_num):
                step = epoch * dataset.train_batch_num + num
                lr = FLAGS.LEARNING_RATE * (0.5 ** (np.floor(epoch / FLAGS.LR_DECAY_EPOCH)))

                # Train model
                train_x, train_y = dataset.get_train_batch()
                loss, summary, _ = sess.run([cross_entropy, merged, train_step],
                                            feed_dict={model._inputs:train_x,
                                                       label:train_y,
                                                       learning_rate:lr})
                # Evaluate on validation set.
                valid_x = dataset.X_valid
                valid_y = dataset.Y_valid
                valid_predict = sess.run(output_class, feed_dict={model._inputs: valid_x})
                valid_eval = FLAGS.Weighted_RMSE(valid_y, valid_predict)


                if FLAGS.SAVE_SUMMARY:
                    writer.add_summary(summary, step)

                # Save model setting
                if step % FLAGS.SAVE_STEP == 0:
                    saver.save(sess, os.path.join(FLAGS.MODEL_PATH, 'model'), global_step=step)

                if valid_eval <= saved_eval:
                    saved_epoch = epoch
                    saved_num = num
                    saved_eval = valid_eval

                # Display setting
                if step % FLAGS.DISPLAY_STEP == 0:
                    rate = (step + 1) * FLAGS.BATCH_SIZE / (time.time() - start_time)
                    remaining = (FLAGS.EPOCH * dataset.train_batch_num - step) * FLAGS.BATCH_SIZE / rate
                    print("###################################################")
                    print("progress  epoch %d  step %d / %d  image/sec %0.1f  remaining %0.1fm" %
                          (epoch, num, dataset.train_batch_num, rate, remaining / 60))
                    print("- Loss =", loss)
                    print("- Weighted RMSE on validation =", valid_eval)
                    print("- Accuracy on validation =", np.sum(valid_y==valid_predict)/np.shape(valid_y)[0])
                    print("- Best(but not saved) Weight RMSE on validation =", saved_eval)
                    print("- Best model on validation at epoch =", saved_epoch, "step =", saved_num)
                    print("- Min kp-index on validation set :", np.min(valid_predict))
                    print("- Max kp-index on validation set :", np.max(valid_predict))

    print("Finish!")


if __name__ == '__main__':
    FLAGS = Config()
    FLAGS.MODE = 'train'
    main(FLAGS)

