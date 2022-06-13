# 入力画像とサイズとパラメータ数
# パラメータとは，重みも含む

# CNNの一番の特徴は，畳み込み層とプーリング層の繰り返し．

# 畳み込み層
# 畳み込み層とは，画像に対して，カーネル（フィルタ）を適用し，画像の特徴量を抽出するような役目を担う層
# 最適化が必要な重みパラメータの数は画像サイズではなくフィルタのサイズに依存するため，MLPとは違い，画像のサイズが大きくなっても，パラメータ数は増大しない．
# また，プーリング層とは，画像を縮小するような層のことで，小さな位置変化に対して頑健にするような役目を担っている．

# https://cs231n.github.io
# ↑スタンフォード大学のCNNの講義
# 入出力が1チャンネル（白黒）だが，実際の畳み込み層では，入出力が複数のチャンネルになることがほとんど
# 畳み込み層の重みパラメータ数は入出力のチャンネル数に依存する．例えば，カラー画像のように入力データが3チャンネルで出直を6チャンネルにしたい場合，
# 重みパラメータは，3*3*3*6（カーネルサイズ ＊ 入力チャンネル数 ＊ 出力チャンネル数）となり，入出力が1チャンネルの畳み込み層の重みパラメータの18倍になる．

# プーリング層
# プーリング層にもいくつか種類がある．
# 最も使われるのはマックスプーリング．
# 入力データを小さな領域に分割し，各領域の最大値をとってくることで，データを縮小する．データが縮小されるので，計算コストが軽減されることに加え，
# 各領域内の位置の違いを無視するため，小さな位置変化に対して頑健なモデルを構築できる．

from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.callbacks import TensorBoard

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# CIFAR-10は32×32×3の大きさのカラー画像なので，最初の層のニューロンへの入力は32×32×3=3072となる．
print("x_train.shape: ", x_train.shape)
print("y_train.shape: ", y_train.shape)
print("x_test.shape: ", x_test.shape)
print("y_test.shape: ", y_test.shape)

x_train = x_train/255.
x_test = x_test/255.

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential()

# filters: 出力のチャンネル数（特徴マップの数）
# kernel_size: カーネルの大きさ
# strides: カーネルをずらす幅
# padding: データの端をどのように取り扱うかを指定する．padding = "same": 入力と出力のサイズを等しくする．padding = "valid" はサイズを等しくしない場合．

model.add(Conv2D(filters=32,
                 input_shape=(32,32,3),
                 kernel_size=(3,3),
                 strides=(1,1),
                 padding="same",
                 activation="relu"))

model.add(Conv2D(filters=32,
                 kernel_size=(3,3),
                 strides=(1,1),
                 padding="same",
                 activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64,
                 kernel_size=(3,3),
                 strides=(1,1),
                 padding="same",
                 activation="relu"))

model.add(Conv2D(filters=64,
                 kernel_size=(3,3),
                 strides=(1,1),
                 padding="same",
                 activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

print(model.output_shape)

model.add(Flatten())

print(model.output_shape)

model.add(Dense(units=512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(units=10, activation="softmax"))

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

tsb = TensorBoard(log_dir="logs/{}")

history_model1 = model.fit(x_train,
                           y_train,
                           batch_size=32,
                           epochs=20,
                           validation_split=0.2,
                           callbacks=[tsb])

