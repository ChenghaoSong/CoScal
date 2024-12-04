# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('/data/tps_hyscal.csv')
data_ad = pd.read_csv('/data/tps_ad.csv')
tps = data['tps']
tps_ad = data_ad['tps']
for i in range(0, 100):
    if tps[i] < 999:
        tps[i] = 999
    if tps_ad[i] < 999:
        tps_ad[i] = 999

tps_csv = np.zeros((2, 100))
tps_csv[0] = tps
tps_csv[1] = tps_ad
tps_csv = tps_csv.T
csv = pd.DataFrame(data=tps_csv)

names = ['tps_hyscal', 'tps_ad']
csv.columns = names
csv['tps_hyscal'] = csv['tps_hyscal'].astype(int)
csv['tps_ad'] = csv['tps_ad'].astype(int)
csv.to_csv('/csv/tps.csv')

plt.xlabel('time')
plt.ylabel('tps')
plt.plot(tps, label='tps', color='red')
plt.plot(tps_ad, label='tps_ad', color='blue')
plt.legend()
plt.savefig('tps_compare.png')
plt.show()
