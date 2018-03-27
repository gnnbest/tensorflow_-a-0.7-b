
import tensorflow as tf
import random
import numpy as np

in_size = 2
out_size = 1
learning_rate = 0.01

x_ = tf.placeholder(dtype=tf.float32, shape=[None, in_size], name='input')
y_ = tf.placeholder(dtype=tf.float32, shape=[None, out_size], name='label')

W = tf.Variable(tf.zeros([in_size, out_size]))
b = tf.Variable(tf.zeros([1, out_size]))

y = tf.matmul(x_, W) + b
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_), reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

init = tf.initialize_all_variables()

session = tf.Session()
session.run(init)

train_batch_size = 40

for i in range(0, 5000):
    batch_a = np.random.uniform(-500.0, 500.0, train_batch_size)
    batch_b = np.random.uniform(-1000, 1000, train_batch_size)

    x_data = np.vstack((batch_a, batch_b)).T
    y_data_tmp = batch_a + 0.7 * batch_b

    y_data = np.reshape(y_data_tmp, [y_data_tmp.shape[0], 1])

    loss_value, _ = session.run(fetches=[loss, train_step], feed_dict={x_:x_data, y_:y_data})

    if(i % 50 == 0):
        print("i = %d, loss = %f " % (i, loss_value))

# test
test_batch_size = 1
for j in range(10):
    a = np.random.uniform(-500, 500, test_batch_size)
    b = np.random.uniform(-1000, 1000, test_batch_size)

    x_data = np.vstack((a, b)).T
    y_data_tmp = a + 0.7 * b
    y_data = np.reshape(y_data_tmp, [y_data_tmp.shape[0], 1])

    prediction = session.run(y, feed_dict={x_:x_data, y_:y_data})

    print('org:', y_data_tmp, 'dec:', prediction)

session.close()







