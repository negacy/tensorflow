import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
size = 1000
vec = []
for i in range(size):
    x = np.random.normal(0.0, 0.55)
    y = x*0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vec.append([x,y])
x_dat = [v[0] for v in vec]
y_dat = [v[1] for v in vec]
#plt.plot(x_dat, y_dat, 'bo')
#plt.show()
#defining linear regression model
W = tf.Variable(tf.random_uniform([1], -1, 1))
b = tf.Variable(tf.zeros([1]))
y = W*x_dat + b
#cost function using square mean error
loss = tf.reduce_mean(tf.square(y-y_dat))
#gradient descent with 0.5 learning rate
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
tf.train.SummaryWriter("./log", sess.graph)
sess.run(init)

for step in range(10):
    sess.run(train)
    print(step, sess.run(W), sess.run(b))
    plt.plot(x_dat, y_dat, 'bo')
    plt.plot(x_dat, sess.run(W)*x_dat + sess.run(b), 'r')
    plt.xlabel('x')
    plt.xlim(-2,2)
    plt.ylim(0.1, 0.6)
    plt.ylabel('y')
    plt.legend()
    plt.show()
