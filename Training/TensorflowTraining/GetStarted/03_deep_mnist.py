# -*- coding: cp-949 -*-
# https://www.tensorflow.org/get_started/mnist/pros


import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


sess = tf.InteractiveSession()


# x:
#    784
# [           ]
# [           ]
# ...
# [           ]
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


# W:
#       10
#      [  ]
# 784  [  ]
#      [  ]
#      ...
#      [  ]
W = tf.Variable(tf.zeros([784,10]))

#>>> W.get_shape()
#TensorShape([Dimension(784), Dimension(10)])
#>>> W.get_shape()[1]
#Dimension(10)
#>>> W.get_shape()[1]  == 10
#True
#>>> W.get_shape()[1]  +1
#Dimension(11)
#>>> int(W.get_shape()[1])

# b:
#    10
# 1 [  ]
b = tf.Variable(tf.zeros([10]))

# Dimension
#  [열]
#   ^
#  [행, 열]
#   ^
#  [깊이, 행, 열]


sess.run(tf.global_variables_initializer())


y = tf.matmul(x,W) + b


# Dimension이 확장되어 연산된다.

#>>> sess.run(b + 1)
#array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.], dtype=float32)
#>>> 
#>>> sess.run((b + 1) * 2) 
#array([ 2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.], dtype=float32)
#>>> W + b
#<tf.Tensor 'add_5:0' shape=(784, 10) dtype=float32>
#>>> sess.run(W + b)
#array([[ 0.,  0.,  0., ...,  0.,  0.,  0.],
#       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
#       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
#       ..., 
#       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
#       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
#       [ 0.,  0.,  0., ...,  0.,  0.,  0.]], dtype=float32)
#>>> sess.run(W + (b+1))
#array([[ 1.,  1.,  1., ...,  1.,  1.,  1.],
#       [ 1.,  1.,  1., ...,  1.,  1.,  1.],
#       [ 1.,  1.,  1., ...,  1.,  1.,  1.],
#       ..., 
#       [ 1.,  1.,  1., ...,  1.,  1.,  1.],
#       [ 1.,  1.,  1., ...,  1.,  1.,  1.],
#       [ 1.,  1.,  1., ...,  1.,  1.,  1.]], dtype=float32)




cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))


# Train the Model

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


for _ in range(1000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})


# Evaluate the Model

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))




# Weight Initialization

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# Convolution and Pooling

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')




#############################
# First Convolution Layer

# 32 features for each 5x5 patch

# shape: [5,5,1,32] means [ width, height, channel, features]
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])


# [-1,28,28,1] means [ , width, height, channel]
x_image = tf.reshape(x, [-1,28,28,1])



# 28x28x1 --> 14x14x32
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


##############################
# Second Convolution Layer

# [5, 5, 32, 64] means [ width, height, channel(previous layer's features), features]
#  width, height = 5,5 because of 'SAME' 
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])


# 14x14x32 --> 7x7x64
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)



#############################
# Densely Connected Layer

# 7x7x64 -> [ 7 * 7 * 64 ] 
# 1024 : nodes
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])


h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)




#############################
# Dropout

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


#############################
# Read out Layer

# output layer
# 1024: node number of previous layer
# 10: 0~9 number digit
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


#############################
# Train and Evaluate the Model
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step= tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_:batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g" % accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
