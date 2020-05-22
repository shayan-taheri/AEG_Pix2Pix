from keras.datasets import cifar10
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

for ix in range(0,len(x_train)):
    fig = plt.figure()
    plt.imshow(x_train[ix])
    print(y_train[ix])
    plt.savefig('/home/shayan/CIFAR/Train/' + str(ix) + '.png')
    plt.close(fig)
    np.savetxt('/home/shayan/CIFAR/Train_Label/' + str(ix) + '.txt',y_train[ix])

for ix in range(0,len(x_test)):
    fig = plt.figure()
    plt.imshow(x_test[ix])
    plt.savefig('/home/shayan/CIFAR/Test/' + str(ix) + '.png')
    plt.close(fig)
    np.savetxt('/home/shayan/CIFAR/Test_Label/' + str(ix) + '.txt',y_test[ix])
