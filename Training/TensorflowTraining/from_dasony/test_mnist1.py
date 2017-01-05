import tensorflow as tf

# MNIST 데이터 불러오기
from tensorflow.examples.tutorials.mnist import input_data
mnistData = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 28*28 image data features
x = tf.placeholder(tf.float32, [None, 784])

# theta
W = tf.Variable(tf.zeros([784,10]))

# bias term
b = tf.Variable(tf.zeros([10]))

# h(x) using softmax
y = tf.nn.softmax(tf.matmul(x, W) + b)

# prediction values
y_ = tf.placeholder(tf.float32, [None, 10])

# Cost function
cross_entropy = tf.reduce_mean(- tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# Gradient descent
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 모든 변수 초기화
init = tf.initialize_all_variables()

# Session 생성
sess = tf.Session()
sess.run(init)

# training
for i in range(1000):
    batch_xs, batch_ys = mnistData.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})

# training 결과 확인
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnistData.test.images, y_: mnistData.test.labels}))
