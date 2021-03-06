import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)
y_label = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)

m = tf.Variable(0.29220241)
b = tf.Variable(0.84038402)

error = 0

for x,y in zip(x_data,y_label):

    y_hat = m*x + b

    error += (y-y_hat)**2 

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()


with tf.Session() as sess:

    sess.run(init)

    epochs = 1

    for i in range(epochs):

        _ , final_slope , final_intercept = sess.run([train,m,b])


    # Fetch Back Results
    print(final_slope,final_intercept)
    #final_slope , final_intercept = sess.run([m,b])
    #print(final_slope,final_intercept)

X_test = np.linspace(-1,11,10)


# y= mx + b

y_pred_plot = final_slope * X_test + final_intercept



plt.plot(X_test,y_pred_plot , 'r')

plt.plot(x_data , y_label , "*")

plt.show()

