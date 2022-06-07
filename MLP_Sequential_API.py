from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.callbacks import TensorBoard


(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("x_train.shape: ", x_train.shape)
print("y_train.shape: ", y_train.shape)
print("x_test.shape: ", x_test.shape)
print("y_test.shape: ", y_test.shape)

# 60000 * 28 * 28 -> 60000 * 784
x_train = x_train.reshape(60000, 784)
x_train = x_train/255
x_test = x_test.reshape(10000, 784)
x_test = x_test/255

# 60000個の要素ベクトルを持つベクトルを60000*10の行列にする．このベクトルを「1-hotベクトル」と呼ぶ
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Sequential API: Kerasでモデルを構築する手法の一つ．Sequential APIでは，用意されているレイヤーをAddメソッドで追加していくだけで，簡単にモデルが構築可能
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

# 中間層: reluを使用すると収束が早くなる場合があることが知られている
model.add(Dense(units=64, input_shape=(784,), activation="relu"))
model.add(Dense(units=10, activation="softmax"))

# 交叉エントロピー: 交叉エントロピーとは二つの確率分布の間に定義される尺度．分類問題では，この値が小さくなるように学習する．
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])


tsb = TensorBoard(log_dir="logs/{}")

history_adam = model.fit(x_train,
                         y_train,
                         batch_size=32,
                         epochs=20,
                         validation_split=.2,
                         callbacks=[tsb])

## tensorboard --logdir=logs/




