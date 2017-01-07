import tensorflow as tf

a = tf.placeholder('float')
b = tf.placeholder('float')

y = tf.mul(a,b)

with tf.Session() as sess:
    writer = tf.train.SummaryWriter("./log", sess.graph)
    print(sess.run(y, feed_dict={a:2, b:3}))

