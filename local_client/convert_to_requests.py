import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

dataset = pd.read_csv('/data/client_cpu_usage.csv')
client = dataset['client']
cpu = dataset['total cpu']
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="elu"),
    tf.keras.layers.Dense(8, activation="elu"),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse"
)

client = client.values
cpu = cpu.values

history = model.fit(cpu, client, epochs=20000)
x_predict = model.predict(cpu)

plt.plot(cpu, client, 'b')
plt.plot(cpu, x_predict, 'r')
plt.show()

cpu_info = pd.read_csv('/data/Machine_usage_groupby.csv')
cpu_Alibaba = cpu_info['cpu_util_percent']
cpu_Alibaba = cpu_Alibaba.values

requests_Alibaba = model.predict(cpu_Alibaba)
plt.plot(requests_Alibaba, cpu_Alibaba)
plt.show()

requests = requests_Alibaba
csv = pd.DataFrame(data=requests)
names = ['requests']
csv.columns = names
csv.to_csv('/data/Alibaba_requests.csv')

