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
for image in glob.glob('/home/shayan/Proposal_Pix2Pix/Data/Training/Fooling_Clean_Data/*'):
    im = cv2.imread(image)
    X_train.append(im)
    y_train.append(int(image[len(image) - 6]))

print(y_train)

X_test = []
y_test = []
for image in glob.glob('/home/shayan/Proposal_Pix2Pix/Data/Testing/Fooling_Clean_Data/*'):
    im = cv2.imread(image)
    X_test.append(im)
    y_test.append(int(image[len(image) - 6]))

print(y_test)

X_test = np.array(X_test)
X_train = np.reshape(X_train, [-1, img_size, img_size, img_chan])
X_train = X_train.astype(np.float32) / 255
X_test = np.reshape(X_test, [-1, img_size, img_size, img_chan])
X_test = X_test.astype(np.float32) / 255

print(X_test.shape)
print(len(y_test))

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


len_list = []
for img in glob.glob('/home/shayan/Proposal_Pix2Pix/Data/Testing/Testing_Out/Epoch_3/outputs/' + '*.png'):
    len_list.append(len(img))

# Finding the unique number of elements:
Slen_list = np.unique(len_list)

print(Slen_list)

DVec1 = []
DVec2 = []
DVec3 = []
DVec4 = []
for img in glob.glob('/home/shayan/Proposal_Pix2Pix/Data/Testing/Testing_Out/Epoch_3/outputs/' + '*.png'):
    if (len(img) == Slen_list[0]):
        DVec1.append(img)
    if (len(img) == Slen_list[1]):
        DVec2.append(img)
    if (len(img) == Slen_list[2]):
        DVec3.append(img)

DVec_Sort = []
DVec_Sort.extend(sorted(DVec1))
DVec_Sort.extend(sorted(DVec2))
DVec_Sort.extend(sorted(DVec3))

YAdv = []
for it in range(0,len(DVec_Sort)):
    name_temp = DVec_Sort[it]
    YAdv.append(int(name_temp[len(name_temp)-14]))

print(YAdv)

'''
Index = []
for ld in range(0,len(DVec_Sort)):
    name_temp = DVec_Sort[ld]
    print(name_temp[40:45])
    if (name_temp[45] == '_'):
        # print(name_temp[44:45]) # The last one is ignored!
        Index.append(name_temp[44:45])
    if (name_temp[46] == '_'):
        Index.append(name_temp[44:46])
    if (name_temp[47] == '_'):
        Index.append(name_temp[44:47])
    if (name_temp[48] == '_'):
        Index.append(name_temp[44:48])

print(Index)
'''

import cv2
from PIL import Image

XAdv_Data = []
for ir in range(0,len(DVec_Sort)):
    im = Image.open(DVec_Sort[ir])
    im = im.resize((128, 128))
    im = np.array(im)
    XAdv_Data.append(im)

XAdv_Data = np.array(XAdv_Data)

XAdv_Data = np.reshape(XAdv_Data, [-1, img_size, img_size, img_chan])
XAdv_Data = XAdv_Data.astype(np.float32)/ 255

X_comb = np.concatenate((X_train, XAdv_Data))
Y_comb = np.concatenate((y_train, YAdv))

print(X_comb.shape)
print(len(Y_comb))

to_categorical = tf.keras.utils.to_categorical
Y_comb = to_categorical(Y_comb)

y_test.append(1)
y_test = to_categorical(y_test)
y_test = y_test[0:len(y_test)-1]

print(y_test)

print('\nTraining')

train(sess, env, X_comb, Y_comb, load=False, epochs=5,
      name='botnet_retraining')

for i in range(0,len(X_test)):

    xorg_ini, y0_ini = X_test[i], y_test[i]

    xorg = np.expand_dims(xorg_ini, axis=0)
    y0 = np.expand_dims(y0_ini, axis=0)

    xadvs_2 = make_jsma(sess, env, xorg, eps=5, epochs=1)

    print('\nEvaluating on Single JSMA adversarial data')
    jsma_loss, jsma_acc = evaluate(sess, env, xadvs_2, y0)

    y0 = y0_ini
    xorg = np.squeeze(xorg, axis=0)

    xadvs_2 = [xorg] + xadvs_2
    
    xadvs_2 = np.squeeze(xadvs_2)

    # JSMA Attack
    if jsma_acc <= 0.20:
        fig = plt.figure()
        plt.imshow(xadvs_2)
        plt.imsave('/home/shayan/Proposal_Pix2Pix/Data/Retraining_Results/Epoch_3/Fooling_Data/xadvs_' + str(i) + '_jsma_' + '_' + str(jsma_acc) + '_[' + str((int(y0[0])*0+int(y0[1])*1)) + '].png',xadvs_2)
        plt.close(fig)
        fig = plt.figure()
        plt.imshow(xorg)
        plt.imsave('/home/shayan/Proposal_Pix2Pix/Data/Retraining_Results/Epoch_3/Fooling_Clean_Data/xadvs_' + str(i) + '_jsma_' + '_' + str(jsma_acc) + '_[' + str((int(y0[0])*0+int(y0[1])*1)) + '].png',xorg)
        plt.close(fig)
    else:
        fig = plt.figure()
        plt.imshow(xadvs_2)
        plt.imsave('/home/shayan/Proposal_Pix2Pix/Data/Retraining_Results/Epoch_3/Unfooling_Data/xadvs_' + str(i) + '_jsma_' + '_' + str(jsma_acc) + '_[' + str((int(y0[0])*0+int(y0[1])*1)) + '].png',xadvs_2)
        plt.close(fig)
        fig = plt.figure()
        plt.imshow(xorg)
        plt.imsave('/home/shayan/Proposal_Pix2Pix/Data/Retraining_Results/Epoch_3/Unfooling_Clean_Data/xadvs_' + str(i) + '_jsma_' + '_' + str(jsma_acc) + '_[' + str((int(y0[0])*0+int(y0[1])*1)) + '].png',xorg)
        plt.close(fig)