import pandas as pd
import numpy as np

dataset = pd.read_csv('/jmeter_rawdata/jmeter_rawdata_58.csv')

data = dataset['latency']
data_avg = np.mean(data)
print(data_avg)