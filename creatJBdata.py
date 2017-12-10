import pickle
import numpy as np
import tensorflow as tf


def load_jbdata():
    with open('data/JBtestDataF.pkl', 'rb') as f:
        testX1 = pickle.load(f)
        testX2 = pickle.load(f)
        testY = pickle.load(f)
        return testX1, testX2,testY


if __name__ == '__main__':
    testX1, testX2, label = load_jbdata()
    print(testX1.shape)
    print(testX2.shape)
    print(label.shape)
    # testX1, testX2,label = load_jbdata()
    # print(testX1.shape)
    # print(testX2.shape)
    # print(label.shape)
    #
    # saver = tf.train.import_meta_graph("checkpoint/25000.ckpt.meta")
    # with tf.Session() as sess:
    #     input = tf.get_default_graph().get_tensor_by_name('input/x:0')
    #     validate = tf.get_default_graph().get_tensor_by_name('DeepID1/validate:0')
    #     saver.restore(sess, 'checkpoint/15000.ckpt')
    #     data1 = sess.run(validate, {input: testX1})
    #     data2 = sess.run(validate, {input: testX2})
    # with open('data/JBtestDataF.pkl', 'wb') as f:
    #     pickle.dump(data1, f, pickle.HIGHEST_PROTOCOL)
    #     pickle.dump(data2, f, pickle.HIGHEST_PROTOCOL)
    #     pickle.dump(label, f, pickle.HIGHEST_PROTOCOL)

