import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(len(x_train))
print(len(x_test))
print(len(y_train))
print(len(y_test))

print(x_train[0])

plt.imshow(x_train[0], cmap='gray')
plt.show()

print(y_train[0])

plt.imshow(x_test[0], cmap='gray')
plt.show()
