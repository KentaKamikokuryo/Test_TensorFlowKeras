import tensorflow as tf

a = tf.constant(1, name="a")
b = tf.constant(1, name="b")

c = a+b

print(c)

with tf.Session() as sess:
    print(sess.run(c))