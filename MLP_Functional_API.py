from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.callbacks import TensorBoard
from keras.layers import Input, Dense
from keras.models import Model

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)/255.
x_test = x_test.reshape(60000, 784)/255.

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

tsb = TensorBoard(log_dir="logs/{}")

input = Input(shape=(784, ))
middle = Dense(units=64, activation="relu")(input)
output = Dense(units=10, activation="softmax")(middle)
model = Model(inputs = [input], outputs = [output])

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.fit(x_train,
          y_train,
          batch_size=32,
          epochs=20,
          callbacks=[tsb],
          validation_split=.2)