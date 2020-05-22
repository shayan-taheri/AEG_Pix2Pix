import tensorflow as tf


def create_adversarial(x):

    layers = []

    with tf.variable_scope('conv0'):
        z = tf.layers.conv2d(x, filters=32, kernel_size=[3, 3],
                             padding='same', activation=tf.nn.relu)
        layers.append(z)

        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)
        layers.append(z)

    with tf.variable_scope('conv1'):
        z = tf.layers.conv2d(z, filters=64, kernel_size=[3, 3],
                             padding='same', activation=tf.nn.relu)
        layers.append(z)

        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    with tf.variable_scope('flatten'):
        shape = z.get_shape().as_list()
        z = tf.reshape(z, [-1, np.prod(shape[1:])])
        layers.append(z)

    with tf.variable_scope('mlp'):
        z = tf.layers.dense(z, units=128, activation=tf.nn.relu)
        layers.append(z)

        z = tf.layers.dropout(z, rate=0.25, training=training)
        layers.append(z)

    logits_ = tf.layers.dense(z, units=10, name='logits')
    y = tf.nn.softmax(logits_, name='ybar')
    layers.append(y)

    return layers[-1]










