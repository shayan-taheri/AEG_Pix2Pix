import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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

print('\nLoading Dataset')

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


def train(sess, env, X_data, y_data, X_valid=None, y_valid=None, epochs=1,
          load=False, shuffle=True, batch_size=128, name='model'):
    """
    Train a TF model by running env.train_op.
    """
    if load:
        if not hasattr(env, 'saver'):
            print('\nError: cannot find saver op')
            return
        print('\nLoading saved model')
        return env.saver.restore(sess, 'model/{}'.format(name))

    print('\nTrain model')
    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    for epoch in range(epochs):
        print('\nEpoch {0}/{1}'.format(epoch + 1, epochs))

        if shuffle:
            print('\nShuffling data')
            ind = np.arange(n_sample)
            np.random.shuffle(ind)
            print(ind)
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
        env.saver.save(sess, 'model/{}'.format(name))


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
print('\nTraining')

train(sess, env, X_train, y_train, load=True, epochs=5,
      name='BotnetData')

'''
for i in range(6413,6414):

    xorg_ini, y0_ini = X_test[i], y_test[i]

    xorg = np.expand_dims(xorg_ini, axis=0)
    y0 = np.expand_dims(y0_ini, axis=0)

    xadvs_0 = make_fgsm(sess, env, xorg, eps=0.1, epochs=1)
    xadvs_1 = make_jsma(sess, env, xorg, eps=5, epochs=1)
    xadvs_2 = make_deepfool(sess, env, xorg, eps=5, epochs=1)
    xadvs_3 = make_cw(sess, env, xorg, eps=100, epochs=1)

    print('\nEvaluating on Single FGSM adversarial data')
    fgsm_loss, fgsm_acc = evaluate(sess, env, xadvs_0, y0)
    print('\nEvaluating on Single JSMA adversarial data')
    jsma_loss, jsma_acc = evaluate(sess, env, xadvs_1, y0)
    print('\nEvaluating on Single DeepFool adversarial data')
    deep_loss, deep_acc = evaluate(sess, env, xadvs_2, y0)
    print('\nEvaluating on Single CW adversarial data')
    cw_loss, cw_acc = evaluate(sess, env, xadvs_3, y0)

    y0 = y0_ini
    xorg = np.squeeze(xorg, axis=0)

    xadvs_0 = [xorg] + xadvs_0
    xadvs_1 = [xorg] + xadvs_1
    xadvs_2 = [xorg] + xadvs_2
    xadvs_3 = [xorg] + xadvs_3

    xadvs_0 = np.squeeze(xadvs_0)
    xadvs_1 = np.squeeze(xadvs_1)
    xadvs_2 = np.squeeze(xadvs_2)
    xadvs_3 = np.squeeze(xadvs_3)

    # FGSM Attack
    if fgsm_acc <= 0.20:
        fig = plt.figure()
        plt.imshow(xadvs_0)
        plt.imsave('/home/shayan/Adv_Dataset/Test/FGSM/Fooling_Data/xadvs_' + str(i) + '_fgsm_' + '_' + str(fgsm_acc) + '_[' + str((int(y0[0])*0+int(y0[1])*1)) + '].png',xadvs_0)
        plt.close(fig)
        fig = plt.figure()
        plt.imshow(xorg)
        plt.imsave('/home/shayan/Adv_Dataset/Test/FGSM/Fooling_Clean_Data/xadvs_' + str(i) + '_fgsm_' + '_' + str(fgsm_acc) + '_[' + str((int(y0[0])*0+int(y0[1])*1)) + '].png',xorg)
        plt.close(fig)
    else:
        fig = plt.figure()
        plt.imshow(xadvs_0)
        plt.imsave('/home/shayan/Adv_Dataset/Test/FGSM/Unfooling_Data/xadvs_' + str(i) + '_fgsm_' + '_' + str(fgsm_acc) + '_[' + str((int(y0[0])*0+int(y0[1])*1)) + '].png',xadvs_0)
        plt.close(fig)
        fig = plt.figure()
        plt.imshow(xorg)
        plt.imsave('/home/shayan/Adv_Dataset/Test/FGSM/Unfooling_Clean_Data/xadvs_' + str(i) + '_fgsm_' + '_' + str(fgsm_acc) + '_[' + str((int(y0[0])*0+int(y0[1])*1)) + '].png',xorg)
        plt.close(fig)

    # JSMA Attack
    if jsma_acc <= 0.20:
        fig = plt.figure()
        plt.imshow(xadvs_1)
        plt.imsave('/home/shayan/Adv_Dataset/Test/JSMA/Fooling_Data/xadvs_' + str(i) + '_jsma_' + '_' + str(jsma_acc) + '_[' + str((int(y0[0])*0+int(y0[1])*1)) + '].png',xadvs_1)
        plt.close(fig)
        fig = plt.figure()
        plt.imshow(xorg)
        plt.imsave('/home/shayan/Adv_Dataset/Test/JSMA/Fooling_Clean_Data/xadvs_' + str(i) + '_jsma_' + '_' + str(jsma_acc) + '_[' + str((int(y0[0])*0+int(y0[1])*1)) + '].png',xorg)
        plt.close(fig)
    else:
        fig = plt.figure()
        plt.imshow(xadvs_1)
        plt.imsave('/home/shayan/Adv_Dataset/Test/JSMA/Unfooling_Data/xadvs_' + str(i) + '_jsma_' + '_' + str(jsma_acc) + '_[' + str((int(y0[0])*0+int(y0[1])*1)) + '].png',xadvs_1)
        plt.close(fig)
        fig = plt.figure()
        plt.imshow(xorg)
        plt.imsave('/home/shayan/Adv_Dataset/Test/JSMA/Unfooling_Clean_Data/xadvs_' + str(i) + '_jsma_' + '_' + str(jsma_acc) + '_[' + str((int(y0[0])*0+int(y0[1])*1)) + '].png',xorg)
        plt.close(fig)

    # DeepFool Attack
    if deep_acc <= 0.20:
        fig = plt.figure()
        plt.imshow(xadvs_2)
        plt.imsave('/home/shayan/Adv_Dataset/Test/DeepFool/Fooling_Data/xadvs_' + str(i) + '_deepfool_' + '_' + str(deep_acc) + '_[' + str((int(y0[0])*0+int(y0[1])*1)) + '].png',xadvs_2)
        plt.close(fig)
        fig = plt.figure()
        plt.imshow(xorg)
        plt.imsave('/home/shayan/Adv_Dataset/Test/DeepFool/Fooling_Clean_Data/xadvs_' + str(i) + '_deepfool_' + '_' + str(deep_acc) + '_[' + str((int(y0[0])*0+int(y0[1])*1)) + '].png',xorg)
        plt.close(fig)
    else:
        fig = plt.figure()
        plt.imshow(xadvs_2)
        plt.imsave('/home/shayan/Adv_Dataset/Test/DeepFool/Unfooling_Data/xadvs_' + str(i) + '_deepfool_' + '_' + str(deep_acc) + '_[' + str((int(y0[0])*0+int(y0[1])*1)) + '].png',xadvs_2)
        plt.close(fig)
        fig = plt.figure()
        plt.imshow(xorg)
        plt.imsave('/home/shayan/Adv_Dataset/Test/DeepFool/Unfooling_Clean_Data/xadvs_' + str(i) + '_deepfool_' + '_' + str(deep_acc) + '_[' + str((int(y0[0])*0+int(y0[1])*1)) + '].png',xorg)
        plt.close(fig)

    # CW Attack
    if cw_acc <= 0.20:
        fig = plt.figure()
        plt.imshow(xadvs_3)
        plt.imsave('/home/shayan/Adv_Dataset/Test/CW/Fooling_Data/xadvs_' + str(i) + '_cw_' + '_' + str(cw_acc) + '_[' + str((int(y0[0])*0+int(y0[1])*1)) + '].png',xadvs_3)
        plt.close(fig)
        fig = plt.figure()
        plt.imshow(xorg)
        plt.imsave('/home/shayan/Adv_Dataset/Test/CW//Fooling_Clean_Data/xadvs_' + str(i) + '_cw_' + '_' + str(cw_acc) + '_[' + str((int(y0[0])*0+int(y0[1])*1)) + '].png',xorg)
        plt.close(fig)
    else:
        fig = plt.figure()
        plt.imshow(xadvs_3)
        plt.imsave('/home/shayan/Adv_Dataset/Test/CW/Unfooling_Data/xadvs_' + str(i) + '_cw_' + '_' + str(cw_acc) + '_[' + str((int(y0[0])*0+int(y0[1])*1)) + '].png',xadvs_3)
        plt.close(fig)
        fig = plt.figure()
        plt.imshow(xorg)
        plt.imsave('/home/shayan/Adv_Dataset/Test/CW/Unfooling_Clean_Data/xadvs_' + str(i) + '_cw_' + '_' + str(cw_acc) + '_[' + str((int(y0[0])*0+int(y0[1])*1)) + '].png',xorg)
'''