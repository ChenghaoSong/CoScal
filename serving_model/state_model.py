import tensorflow as tf
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt


dataset = pd.read_csv('/data/state_preprocessed.csv', index_col=0)

dataset_columns = dataset.columns
values = dataset.values

# 学习与检测数据的划分
n_train_hours = 80
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]

# 监督学习结果划分
train_x, train_y = train[:, :-1], train[:, -1]
test_x, test_y = test[:, :-1], test[:, -1]

# print(train_x.shape)
# print(train_y.shape)
# print(test_x.shape)
# print(test_y.shape)

model = Sequential()
model.add(Dense(16, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1))

model.compile(loss=tf.keras.losses.Huber(),
              optimizer='adam',
              metrics=["mse"])

history = model.fit(
    train_x,
    train_y,
    epochs=200,
    validation_data=(
        test_x,
        test_y),
    verbose=0)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# test_y_p = model.predict(test_x)
# val = []
# for i in range(0, 131):
#     k = test_y_p[i] - test_y[i]
#     val.append(k)
# plt.plot(test_x, val)
# plt.show()

# print(test_x.shape)
# k = [[4, 4, 4, 1, 4]]
# k = np.array(k)
# print(k.shape)
# K = model.predict(k)
# print(K)
model.save('/tf_model/1')
