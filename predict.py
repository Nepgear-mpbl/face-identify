import tensorflow as tf
import pickle
from PIL import Image
import numpy as np
from scipy.spatial.distance import cosine


def crop_img_by_half_center(src_file_path,dest_file_path):
    im = Image.open(src_file_path)
    x_size, y_size = im.size
    start_point_xy = x_size / 4
    end_point_xy = x_size / 4 + x_size / 2
    box = (start_point_xy, start_point_xy, end_point_xy, end_point_xy)
    new_im = im.crop(box)
    new_new_im = new_im.resize((47, 55))
    new_new_im.save(dest_file_path)
    return new_new_im

def Verify(A, G, x1, x2):
    x1.shape = (-1, 1)
    x2.shape = (-1, 1)
    ratio = np.dot(np.dot(np.transpose(x1), A), x1) + np.dot(np.dot(np.transpose(x2), A), x2) - 2 * np.dot(
        np.dot(np.transpose(x1), G), x2)
    return float(ratio)

def load_jbdata():
    with open('data/JBtestData.pkl', 'rb') as f:
        testX1 = pickle.load(f)
        testX2 = pickle.load(f)
        testY = pickle.load(f)
        return testX1, testX2,testY

if __name__ == '__main__':
    # pic1 = crop_img_by_half_center('testpic/test1.jpg','testpic/test1c.jpg')
    # pic2 = crop_img_by_half_center('testpic/test2.jpg','testpic/test2c.jpg')
    # pic1arr = np.reshape(np.asarray(pic1, dtype='float32'), [1, 55, 47, 3])
    # pic2arr = np.reshape(np.asarray(pic2, dtype='float32'), [1, 55, 47, 3])
    testX1, testX2, label = load_jbdata()

    saver = tf.train.import_meta_graph("checkpoint/25000.ckpt.meta")
    with tf.Session() as sess:
        input = tf.get_default_graph().get_tensor_by_name('input/x:0')
        validate = tf.get_default_graph().get_tensor_by_name('DeepID1/validate:0')
        saver.restore(sess, 'checkpoint/25000.ckpt')
        h1 = sess.run(validate, {input: testX1})
        h2 = sess.run(validate, {input: testX2})
    with open('JointBayesian_Model/A.pkl', 'rb') as f:
        A = pickle.load(f)
    with open('JointBayesian_Model/G.pkl', 'rb') as f:
        G = pickle.load(f)
    # pre_y = np.array([cosine(x, y) for x, y in zip(h1, h2)])
    # print(pre_y)
    predict=np.zeros([len(label)])
    print(len(label))
    count=0
    for i in range(len(label)):
        result = Verify(A, G, h1[i], h2[i])
        if result>=-17.65:
            predict[i]=1
        else:
            predict[i]=0
        if predict[i]==label[i]:
            count+=1
    correct_rate=float(count/len(label))
    print(count)
    print(correct_rate)

