# -*- coding: cp-949 -*-

# https://www.tensorflow.org/get_started/get_started

import tensorflow as tf


# constant
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
print(node1, node2)

sess = tf.Session()
print(sess.run([node1, node2]))


node3 = tf.add(node1, node2)
print("node3: ", node3)
print("sess.run(node3): ", sess.run(node3))




# TensorBoard




# placeholders
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b

print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1,3], b: [2,4]}))


add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 3, b: 4.5}))



# Variables and Model
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

# initialize variable
init = tf.global_variables_initializer()
sess.run(init)


print(sess.run(linear_model, {x: [1,2,3,4]}))



# loss function
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))



# Variable and assign
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))




# tf.train API

# optimizers
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init)
for i in range(1000):
    sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})

print(sess.run([W, b]))







#======================================================
# Complete program

import numpy as np
import tensorflow as tf

# Model parameters
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)

# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(linear_model - y))

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = [1,2,3,4]
y_train = [0, -1, -2, -3]

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(1000):
    sess.run(train, {x:x_train, y:y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train})
print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))

#
#==========================================================




### tf.contrib.learn

##import tensorflow as tf
##import numpy as np

##features = [tf.contrib.layers.real_valued_column("x", dimension=1)]
##estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

##x = np.array([1., 2., 3., 4.])
##y = np.array([0., -1., -2., -3.])
##input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x}, y, batch_size=4, num_epochs=1000)
###AttributeError: module 'tensorflow.contrib.learn.python.learn.learn_io' has no attribute 'numpy_input_fn'
###input_fn = tf.contrib.learn.io.pandas_input_fn({"x":x}, y, batch_size=4, num_epochs=1000)

##estimator.fit(input_fn=input_fn, steps=1000)

##estimator.evaluate(input_fn=input_fn)

