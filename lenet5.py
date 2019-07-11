import numpy as np
import tensorflow as tf
from tensorflow import keras as tfk

mnist = tf.keras.datasets.mnist

(data_train, label_train), (data_test, label_test) = mnist.load_data()
# # # Normalize 0 to 255 value matrix to 0 to 1 value matrix
# data_train, data_test = data_train/255.0, data_test/255.0
train_data = data_train.reshape(data_train.shape[0], 28, 28, 1)
train_label = np.asarray(label_train, dtype=np.int32)
test_data = data_test.reshape(data_test.shape[0], 28, 28, 1)
test_label = np.asarray(label_test, dtype=np.int32)

model = tfk.models.Sequential([
    tfk.layers.Conv2D(filters=6, kernel_size=(5, 5), activation=tf.nn.relu, input_shape=(28, 28, 1)),
    tfk.layers.MaxPool2D(pool_size=(2, 2), strides=2),
    tfk.layers.Conv2D(filters=16, kernel_size=(5, 5), activation=tf.nn.relu),
    tfk.layers.MaxPool2D(pool_size=(2, 2), strides=2),
    tfk.layers.Flatten(),
    tfk.layers.Dense(120, activation=tf.nn.relu),
    tfk.layers.Dense(84, activation=tf.nn.relu),
    tfk.layers.Dense(10, activation=tf.nn.softmax),
])

model.compile(optimizer=tfk.optimizers.Adadelta(), loss=tfk.losses.sparse_categorical_crossentropy, metrics=[tfk.metrics.sparse_categorical_accuracy])

if __name__ == '__main__':

    model.fit(train_data, train_label, epochs=50)
    model.evaluate(test_data, test_label)

