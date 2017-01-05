import tensorflow as tf

x = tf.constant(3)
print(x)

sess = tf.Session()
result = sess.run(x)
print(result)


#-------------------

x = tf.constant(3.)
print(x)

sess = tf.Session()
result = sess.run(x)
print(result)

#-------------------

x = tf.constant([3.,3.])
print(x)

sess = tf.Session()
result = sess.run(x)
print(result)



x = tf.constant([[3.,3.],[5.,5.]])
print(x)

sess = tf.Session()
result = sess.run(x)
print(result)




x = tf.constant([[3.,3.],[5.,5.],[7.,7.]])
print(x)

sess = tf.Session()
result = sess.run(x)
print(result)



#-----------------------------

x = tf.constant( [ [[3.,3.],[5.,5.],[7.,7.]],[[2.,2.],[4.,4.],[6.,6.]] ] )
print(x)

sess = tf.Session()
result = sess.run(x)
print(result)


