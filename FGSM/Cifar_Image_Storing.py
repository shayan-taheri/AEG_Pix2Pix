from keras.datasets import cifar10
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
import scipy
from PIL import Image

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

ix = 0
while ix <= len(x_train):
    fig = plt.figure()
    plt.imshow(x_train[ix])
    plt.imsave('/home/shayan/CIFAR/Train/C' + str(ix) + str(y_train[ix]) + '.png', x_train[ix])
    plt.close(fig)
    np.savetxt('/home/shayan/CIFAR/Train_Label/' + str(ix) + '.txt',y_train[ix])
    ix = ix + 1

iy = 0
while ix <= len(x_train):
    fig = plt.figure()
    plt.imshow(x_test[ix])
    plt.imsave('/home/shayan/CIFAR/Test/C' + str(ix) + str(y_test[ix]) + '.png', x_test[ix])
    plt.close(fig)
    np.savetxt('/home/shayan/CIFAR/Test_Label/' + str(ix) + '.txt',y_test[ix])
    iy = iy + 1
