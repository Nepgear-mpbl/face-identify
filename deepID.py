import tensorflow as tf
import numpy as np
from vec import load_data

testX1, testX2, testY, validX, validY, trainX, trainY = load_data()
class_num = np.max(trainY) + 1


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.zeros(shape))


def Wx_plus_b(weights, x, biases):
    return tf.matmul(x, weights) + biases


# 全连接层
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        weights = weight_variable([input_dim, output_dim])
        biases = bias_variable([output_dim])
        pre_activate = Wx_plus_b(weights, input_tensor, biases)
        if act is not None:
            activations = act(pre_activate, name='activation')
            return activations
        else:
            return pre_activate


# 卷积池化层
def conv_pool_layer(x, w_shape, b_shape, layer_name, act=tf.nn.relu, only_conv=False):
    with tf.name_scope(layer_name):
        W = weight_variable(w_shape)  # 卷积核
        b = bias_variable(b_shape)
        conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
        h = conv + b
        relu = act(h)
        if only_conv is True:
            return relu
        # 2x2最大池化
        pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        return pool


# 预测准确度
def accuracy(y_estimate, y_real):
    correct_prediction = tf.equal(tf.argmax(y_estimate, 1), tf.argmax(y_real, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    return accuracy


def train_step(loss):
    with tf.name_scope('train'):
        return tf.train.AdamOptimizer(1e-4).minimize(loss)


with tf.name_scope('input'):
    h0 = tf.placeholder(tf.float32, [None, 55, 47, 3], name='x')
    y_ = tf.placeholder(tf.float32, [None, class_num], name='y')

h1 = conv_pool_layer(h0, [4, 4, 3, 20], [20], 'Conv_layer_1')
# h1.shape=[52,44,20]->[26,22,20]
h2 = conv_pool_layer(h1, [3, 3, 20, 40], [40], 'Conv_layer_2')
# h2.shape=[24,20,40]->[12,10,40]
h3 = conv_pool_layer(h2, [3, 3, 40, 60], [60], 'Conv_layer_3')
# h3.shape=[10,8,60]->[5,4,60]
h4 = conv_pool_layer(h3, [2, 2, 60, 80], [80], 'Conv_layer_4', only_conv=True)
# h4.shape=[4,3,80]

with tf.name_scope('DeepID1'):
    h3r = tf.reshape(h3, [-1, 5 * 4 * 60])
    h4r = tf.reshape(h4, [-1, 4 * 3 * 80])
    W1 = weight_variable([5 * 4 * 60, 160])
    W2 = weight_variable([4 * 3 * 80, 160])
    b = bias_variable([160])
    h = tf.matmul(h3r, W1) + tf.matmul(h4r, W2) + b
    h5 = tf.nn.relu(h,name="validate")

with tf.name_scope('loss'):
    y = nn_layer(h5, 160, class_num, 'nn_layer', act=None)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= y,labels= y_))
    tf.summary.scalar('loss', loss)

accuracy = accuracy(y, y_)
train_step = train_step(loss)

merged = tf.summary.merge_all()
saver = tf.train.Saver()

if __name__ == '__main__':
    def get_batch(data_x, data_y, start):
        end = (start + 1024) % data_x.shape[0]
        if start < end:
            return data_x[start:end], data_y[start:end], end
        return np.vstack([data_x[start:], data_x[:end]]), np.vstack([data_y[start:], data_y[:end]]), end


    data_x = trainX
    data_y = (np.arange(class_num) == trainY[:, None]).astype(np.float32)
    validY = (np.arange(class_num) == validY[:, None]).astype(np.float32)

    logdir = 'log'
    if tf.gfile.Exists(logdir):
        tf.gfile.DeleteRecursively(logdir)
    tf.gfile.MakeDirs(logdir)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    ckpt=tf.train.get_checkpoint_state('checkpoint')
    if ckpt and ckpt.model_checkpoint_path:
        print(ckpt.model_checkpoint_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    train_writer = tf.summary.FileWriter(logdir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(logdir + '/test', sess.graph)

    idx = (1024*25001)% data_x.shape[0]
    for i in range(25001,50001):
        batch_x, batch_y, idx = get_batch(data_x, data_y, idx)
        #print(idx)
        summary, _ = sess.run([merged, train_step], {h0: batch_x, y_: batch_y})
        train_writer.add_summary(summary, i)

        if i % 100 == 0:
            summary, accu = sess.run([merged, accuracy], {h0: validX, y_: validY})
            test_writer.add_summary(summary, i)
            print('round %f : %f' % (i / 100, accu))
        if i % 5000 == 0 and i != 0:
            saver.save(sess, 'checkpoint/%05d.ckpt' % i)
            print('saved checkpoint/%05d.ckpt'% i)
