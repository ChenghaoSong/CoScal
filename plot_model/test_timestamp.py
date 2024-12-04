import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

j_rawdata = []
j_rawdata_ad = []

for i in range(0, 307):
    data = pd.read_csv('F:/jmeter_rawdata/jmeter_rawdata_%s.csv' % i)

    data = data['timeStamp']
    lenth = len(data)
    k = data[lenth-1] - data[0]
    j_rawdata_ad.append(k)

k = max(j_rawdata_ad)
print(k)