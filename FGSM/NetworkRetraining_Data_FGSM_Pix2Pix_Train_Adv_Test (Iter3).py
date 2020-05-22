import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#s.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from timeit import default_timer
import tensorflow as tf
import cv2
import glob
from attacks import fgm, jsma, deepfool, cw

img_size = 128
img_chan = 3
n_classes = 2
batch_size = 1

class Timer(object):
    def __init__(self, msg='Starting.....', timer=default_timer, factor=1,
                 fmt="------- elapsed {:.4f}s --------"):
        self.timer = timer
        self.factor = factor
        self.fmt = fmt
        self.end = None
        self.msg = msg
    def __call__(self):
        """
        Return the current time
        """
        return self.timer()
    def __enter__(self):
        """
        Set the start time
        """
        print(self.msg)
        self.start = self()
        return self
    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        Set the end time
        """
        self.end = self()
        print(str(self))
    def __repr__(self):
        return self.fmt.format(self.elapsed)
    @property
    def elapsed(self):
        if self.end is None:
            # if elapsed is called in the context manager scope
            return (self() - self.start) * self.factor
        else:
            # if elapsed is called out of the context manager scope
            return (self.end - self.start) * self.factor

'''
print('\nLoading Original Dataset')


X_train = []
y_train = []
for dir in glob.glob('/home/shayan/Dataset/Botnet/Train/*'):
    for image in glob.glob(dir + '/*'):
        im = cv2.imread(image)
        X_train.append(im)
        y_train.append(0)
for dir in glob.glob('/home/shayan/Dataset/Normal/Train/*'):
    for image in glob.glob(dir + '/*'):
        im = cv2.imread(image)
        X_train.append(im)
        y_train.append(1)

X_train = np.array(X_train)
X_test = []
y_test = []
for dir in glob.glob('/home/shayan/Dataset/Botnet/Test/*'):
    for image in glob.glob(dir + '/*'):
        im = cv2.imread(image)
        X_test.append(im)
        y_test.append(0)
for dir in glob.glob('/home/shayan/Dataset/Normal/Test/*'):
    for image in glob.glob(dir + '/*'):
        im = cv2.imread(image)
        X_test.append(im)
        y_test.append(1)

X_test = np.array(X_test)
X_train = np.reshape(X_train, [-1, img_size, img_size, img_chan])
X_train = X_train.astype(np.float32) / 255
X_test = np.reshape(X_test, [-1, img_size, img_size, img_chan])
X_test = X_test.astype(np.float32) / 255

to_categorical = tf.keras.utils.to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

file = open('/home/shayan/Adv_Dataset/TrainingIndices.txt', 'r')
idx_train = file.read()
file.close()

import re
idx_train_str = re.findall('\d+', idx_train)

idx_train_int = []
for iy in range(0,len(idx_train_str)):
    idx_train_int.append(int(idx_train_str[iy]))

X_train_BAK = X_train
Y_train_BAK = y_train

for iy in range(0,len(idx_train_int)):
    iz = idx_train_int[iy]
    X_train[iy] = X_train_BAK[iz]
    y_train[iy] = Y_train_BAK[iz]

print(X_train.shape)
print(y_train.shape)

X_train = X_train[0:50000]
y_train = y_train[0:50000]

print(X_train.shape)
print(y_train.shape)

file = open('/home/shayan/Adv_Dataset/TestingIndices.txt', 'r')
idx_test = file.read()
file.close()

import re
idx_test_str = re.findall('\d+', idx_test)

idx_test_int = []
for iy in range(0,len(idx_test_str)):
    idx_test_int.append(int(idx_test_str[iy]))

X_test_BAK = X_test
Y_test_BAK = y_test

for iy in range(0,len(idx_test_int)):
    iz = idx_test_int[iy]
    X_test[iy] = X_test_BAK[iz]
    y_test[iy] = Y_test_BAK[iz]

print(X_test.shape)
print(y_test.shape)

X_test = X_test[0:10000]
y_test = y_test[0:10000]

print(X_test.shape)
print(y_test.shape)
'''

print('\nConstruction graph')


def model(x, logits=False, training=False):
    with tf.variable_scope('conv0'):
        z = tf.layers.conv2d(x, filters=32, kernel_size=[3, 3],
                             padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    with tf.variable_scope('conv1'):
        z = tf.layers.conv2d(z, filters=64, kernel_size=[3, 3],
                             padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    with tf.variable_scope('flatten'):
        shape = z.get_shape().as_list()
        z = tf.reshape(z, [-1, np.prod(shape[1:])])

    with tf.variable_scope('mlp'):
        z = tf.layers.dense(z, units=128, activation=tf.nn.relu)
        z = tf.layers.dropout(z, rate=0.25, training=training)

    logits_ = tf.layers.dense(z, units=2, name='logits')
    y = tf.nn.softmax(logits_, name='ybar')

    if logits:
        return y, logits_
    return y

class Dummy:
    pass

env = Dummy()

with tf.variable_scope('model'):
    env.x = tf.placeholder(tf.float32, (None, img_size, img_size, img_chan),
                           name='x')
    env.y = tf.placeholder(tf.float32, (None, n_classes), name='y')
    env.training = tf.placeholder_with_default(False, (), name='mode')

    env.ybar, logits = model(env.x, logits=True, training=env.training)

    with tf.variable_scope('acc'):
        count = tf.equal(tf.argmax(env.y, axis=1), tf.argmax(env.ybar, axis=1))
        env.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')

    with tf.variable_scope('loss'):
        xent = tf.nn.softmax_cross_entropy_with_logits(labels=env.y,
                                                       logits=logits)
        env.loss = tf.reduce_mean(xent, name='loss')

    with tf.variable_scope('train_op'):
        optimizer = tf.train.AdamOptimizer()
        vs = tf.global_variables()
        env.train_op = optimizer.minimize(env.loss, var_list=vs)

    env.saver = tf.train.Saver()

with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
    env.x_fixed = tf.placeholder(
        tf.float32, (batch_size, img_size, img_size, img_chan),
        name='x_fixed')
    env.adv_eps = tf.placeholder(tf.float32, (), name='adv_eps')
    env.adv_epochs = tf.placeholder(tf.int32, (), name='adv_epochs')
    env.adv_y = tf.placeholder(tf.int32, (), name='adv_y')

    env.x_fgsm = fgm(model, env.x, epochs=env.adv_epochs, eps=env.adv_eps)
    env.x_deepfool = deepfool(model, env.x, epochs=env.adv_epochs, batch=True)
    env.x_jsma = jsma(model, env.x, env.adv_y, eps=env.adv_eps,
                      epochs=env.adv_epochs)
    env.cw_train_op, env.x_cw, env.cw_noise = cw(model, env.x_fixed,
                                                 y=env.adv_y, eps=env.adv_eps,
                                                 optimizer=optimizer)

print('\nInitializing graph')

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

def evaluate(sess, env, X_data, y_data, batch_size=128):
    """
    Evaluate TF model by running env.loss and env.acc.
    """
    print('\nEvaluating')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    loss, acc = 0, 0

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch))
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        cnt = end - start
        batch_loss, batch_acc = sess.run(
            [env.loss, env.acc],
            feed_dict={env.x: X_data[start:end],
                       env.y: y_data[start:end]})
        loss += batch_loss * cnt
        acc += batch_acc * cnt
    loss /= n_sample
    acc /= n_sample

    print(' loss: {0:.4f} acc: {1:.4f}'.format(loss, acc))
    return loss, acc

def retrain_train(sess, env, X_data, y_data, X_valid=None, y_valid=None, epochs=1,
          shuffle=True, batch_size=128, name='model'):

    """ind
    Train a TF model by running env.train_op.
    """
    if not hasattr(env, 'saver'):
        print('\nError: cannot find saver op')
        return
    print('\nLoading saved model')
    env.saver.restore(sess, 'iter2_retrain/pix2pix_adv/Epoch_10/{}'.format(name))

    print('\nTrain model')
    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    for epoch in range(epochs):
        print('\nEpoch {0}/{1}'.format(epoch + 1, epochs))

        if shuffle:
            print('\nShuffling data')
            ind = np.arange(n_sample)
            np.random.shuffle(ind)
            print(len(X_data))
            print(len(y_data))
            X_data = X_data[ind]
            y_data = y_data[ind]

        for batch in range(n_batch):
            print(' batch {0}/{1}'.format(batch + 1, n_batch))
            start = batch * batch_size
            end = min(n_sample, start + batch_size)
            sess.run(env.train_op, feed_dict={env.x: X_data[start:end],
                                              env.y: y_data[start:end],
                                              env.training: True})
        if X_valid is not None:
            evaluate(sess, env, X_valid, y_valid)

    if hasattr(env, 'saver'):
        print('\n Saving model')
        env.saver.save(sess, 'iter3_retrain/pix2pix_adv/Epoch_10/{}'.format(name))


def retrain_use(sess, env, X_data, y_data, X_valid=None, y_valid=None, epochs=1,
          shuffle=True, batch_size=128, name='model'):

    """ind
    Train a TF model by running env.train_op.
    """
    if not hasattr(env, 'saver'):
        print('\nError: cannot find saver op')
        return
    print('\nLoading saved model')
    env.saver.restore(sess, 'iter3_retrain/pix2pix_adv/Epoch_10/{}'.format(name))

def predict(sess, env, X_data, batch_size=128):
    """
    Do inference by running env.ybar.
    """
    print('\nPredicting')
    n_classes = env.ybar.get_shape().as_list()[1]

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    yval = np.empty((n_sample, n_classes))

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch))
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        y_batch = sess.run(env.ybar, feed_dict={env.x: X_data[start:end]})
        yval[start:end] = y_batch
    print()
    return yval


def make_fgsm(sess, env, X_data, epochs=1, eps=10, batch_size=128):
    print('\nMaking adversarials via FGSM')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch))
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        feed_dict = {env.x: X_data[start:end], env.adv_eps: eps,
                     env.adv_epochs: epochs}
        adv = sess.run(env.x_fgsm, feed_dict=feed_dict)
        X_adv[start:end] = adv
    print()

    return X_adv


def make_jsma(sess, env, X_data, epochs=1, eps=0.001, batch_size=128):
    print('\nMaking adversarials via JSMA')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch))
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        feed_dict = {
            env.x: X_data[start:end],
            env.adv_y: np.random.choice(n_classes),
            env.adv_epochs: epochs,
            env.adv_eps: eps}
        adv = sess.run(env.x_jsma, feed_dict=feed_dict)
        X_adv[start:end] = adv
    print()

    return X_adv


def make_deepfool(sess, env, X_data, epochs=1, eps=10, batch_size=128):
    print('\nMaking adversarials via DeepFool')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch))
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        feed_dict = {env.x: X_data[start:end], env.adv_epochs: epochs}
        adv = sess.run(env.x_deepfool, feed_dict=feed_dict)
        X_adv[start:end] = adv
    print()

    return X_adv

def make_cw(sess, env, X_data, epochs=1, eps=10, batch_size=1):
    """
    Generate adversarial via CW100 optimization.
    """
    print('\nMaking adversarials via CW')
    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)
    for batch in range(n_batch):
        with Timer('Batch {0}/{1}   '.format(batch + 1, n_batch)):
            end = min(n_sample, (batch+1) * batch_size)
            start = end - batch_size
            feed_dict = {
                env.x_fixed: X_data[start:end],
                env.adv_eps: eps,
                env.adv_y: np.random.choice(n_classes)}
            # reset the noise before every iteration
            sess.run(env.cw_noise.initializer)
            for epoch in range(epochs):
                sess.run(env.cw_train_op, feed_dict=feed_dict)
            xadv = sess.run(env.x_cw, feed_dict=feed_dict)
            X_adv[start:end] = xadv

    return X_adv


print('\nLoading FGSM Training Dataset')


len_list = []
for img in glob.glob('/home/shayan/Proposal_Pix2Pix/Data/Training/FGSM/Iteration_3/Training_Data/Epoch_10/outputs/' + '*.png'):
    len_list.append(len(img))

# Finding the unique number of elements:
Slen_list = np.unique(len_list)

print(Slen_list)

DVec1 = []
DVec2 = []
DVec3 = []
DVec4 = []
DVec5 = []
for img in glob.glob('/home/shayan/Proposal_Pix2Pix/Data/Training/FGSM/Iteration_3/Training_Data/Epoch_10/outputs/' + '*.png'):
    if (len(img) == Slen_list[0]):
        DVec1.append(img)
    if (len(img) == Slen_list[1]):
        DVec2.append(img)
    if (len(img) == Slen_list[2]):
        DVec3.append(img)
    if (len(img) == Slen_list[3]):
        DVec4.append(img)
    if (len(img) == Slen_list[4]):
        DVec5.append(img)

DVec_Sort = []
DVec_Sort.extend(sorted(DVec1))
DVec_Sort.extend(sorted(DVec2))
DVec_Sort.extend(sorted(DVec3))
DVec_Sort.extend(sorted(DVec4))
DVec_Sort.extend(sorted(DVec5))

YAdv_Train = []
for it in range(0,len(DVec_Sort)):
    name_temp = DVec_Sort[it]
    YAdv_Train.append(int(name_temp[len(name_temp)-14]))

print(YAdv_Train)
print(len(YAdv_Train))

to_categorical = tf.keras.utils.to_categorical
YAdv_Train = to_categorical(YAdv_Train)

print(len(YAdv_Train))

DVec5 = []

import cv2
from PIL import Image

XAdv_Data_Train = []
for ir in range(0,len(DVec_Sort)):
    im = cv2.imread(DVec_Sort[ir])
    im = cv2.resize(im, (128,128))
    im = np.array(im)
    XAdv_Data_Train.append(im)

XAdv_Data_Train = np.array(XAdv_Data_Train)

print(XAdv_Data_Train.shape)


XAdv_Data_Train = np.reshape(XAdv_Data_Train, [-1, img_size, img_size, img_chan])
XAdv_Data_Train = XAdv_Data_Train.astype(np.float32)/ 255

print(XAdv_Data_Train.shape)


print('\nLoading FGSM Testing Dataset')


len_list = []
for img in glob.glob('/home/shayan/Adv_Dataset/Test/FGSM/Fooling_Data/' + '*.png'):
    len_list.append(len(img))

# Finding the unique number of elements:
Slen_list = np.unique(len_list)

print(Slen_list)

DVec1 = []
DVec2 = []
DVec3 = []
DVec4 = []
for img in glob.glob('/home/shayan/Adv_Dataset/Test/FGSM/Fooling_Data/' + '*.png'):
    if (len(img) == Slen_list[0]):
        DVec1.append(img)
    if (len(img) == Slen_list[1]):
        DVec2.append(img)
    if (len(img) == Slen_list[2]):
        DVec3.append(img)
    if (len(img) == Slen_list[3]):
        DVec4.append(img)

DVec_Sort = []
DVec_Sort.extend(sorted(DVec1))
DVec_Sort.extend(sorted(DVec2))
DVec_Sort.extend(sorted(DVec3))
DVec_Sort.extend(sorted(DVec4))

YAdv_Test = []
for it in range(0,len(DVec_Sort)):
    name_temp = DVec_Sort[it]
    YAdv_Test.append(int(name_temp[len(name_temp)-6]))

print(YAdv_Test)
print(len(YAdv_Test))

to_categorical = tf.keras.utils.to_categorical
YAdv_Test = to_categorical(YAdv_Test)

print(len(YAdv_Test))

import cv2
from PIL import Image

XAdv_Data_Test = []
for ir in range(0,len(DVec_Sort)):
    im = cv2.imread(DVec_Sort[ir])
    im = np.array(im)
    XAdv_Data_Test.append(im)

XAdv_Data_Test = np.array(XAdv_Data_Test)

print(XAdv_Data_Test.shape)

XAdv_Data_Test = np.reshape(XAdv_Data_Test, [-1, img_size, img_size, img_chan])
XAdv_Data_Test = XAdv_Data_Test.astype(np.float32)/ 255

print(XAdv_Data_Test.shape)

print('\nRetraining -- Train Model')

retrain_train(sess, env, XAdv_Data_Train, YAdv_Train, epochs=5,
      name='BotnetData')

#print('\nRetraining -- Use Model')

#retrain_use(sess, env, XAdv_Data_Train, YAdv_Train, epochs=5,
#      name='BotnetData')

for i in range(0,9969):

    xadvs_0, y0_ini = XAdv_Data_Test[i], YAdv_Test[i]

    xadvs_0 = np.expand_dims(xadvs_0, axis=0)
    y0 = np.expand_dims(y0_ini, axis=0)

    print('\nEvaluating on Single FGSM adversarial data')
    fgsm_loss, fgsm_acc = evaluate(sess, env, xadvs_0, y0)

    y0 = y0_ini

    xadvs_0 = np.squeeze(xadvs_0)

    # FGSM Attack
    if fgsm_acc <= 0.20:
        fig = plt.figure()
        plt.imshow(xadvs_0)
        plt.imsave('/home/shayan/Proposal_Pix2Pix/Data/Training/FGSM/Iteration_3/Retraining_Results_Pix2Pix_Adv/Epoch_10/Fooling_Data/xadvs_' + str(i) + '_fgsm_' + '_' + str(fgsm_acc) + '_[' + str((int(y0[0])*0+int(y0[1])*1)) + '].png',xadvs_0)
        plt.close(fig)
    else:
        fig = plt.figure()
        plt.imshow(xadvs_0)
        plt.imsave('/home/shayan/Proposal_Pix2Pix/Data/Training/FGSM/Iteration_3/Retraining_Results_Pix2Pix_Adv/Epoch_10/Unfooling_Data/xadvs_' + str(i) + '_fgsm_' + '_' + str(fgsm_acc) + '_[' + str((int(y0[0])*0+int(y0[1])*1)) + '].png',xadvs_0)
        plt.close(fig)
