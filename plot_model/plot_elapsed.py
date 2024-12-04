import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

j_rawdata = []
j_rawdata_ad = []
j_rawdata_k8s = []
j_rawdata_hyscal = []
for i in range(200, 300):
    data = pd.read_csv('jmeter_rawdata_onlyver/jmeter_rawdata_%s.csv' % i)
    data_ad = pd.read_csv('jmeter_rawdata_adjust/jmeter_rawdata_%s.csv' % i)
    data_k8s = pd.read_csv('k8s/jmeter_rawdata_%s.csv' % i)
    data_hyscal = pd.read_csv('hyscal/jmeter_rawdata_%s.csv' % i)

    data = data['elapsed']
    data_ad = data_ad['elapsed']
    data_k8s = data_k8s['elapsed']
    data_hyscal = data_hyscal['elapsed']
    data_avg = np.mean(data)
    data_ad_avg = np.mean(data_ad)
    data_k8s_avg = np.mean(data_k8s)
    data_hyscal_avg = np.mean(data_hyscal)

    j_rawdata.append(data_avg)
    j_rawdata_ad.append(data_ad_avg)
    j_rawdata_k8s.append(data_k8s_avg)
    j_rawdata_hyscal.append(data_hyscal_avg)

k = sum(j_rawdata_ad) - sum(j_rawdata)
print(k)
plt.xlabel('time')
plt.ylabel('elapsed')
plt.plot(j_rawdata, label='data', color='red')
plt.plot(j_rawdata_ad, label='data_ad', color='blue')
plt.plot(j_rawdata_k8s, label='k8s', color='green')
plt.plot(j_rawdata_hyscal, label='k8s', color='coral')
plt.legend()
plt.savefig('plotfig/elapsed_compare.png')
j_rawdata = pd.DataFrame(data=j_rawdata)
j_rawdata_ad = pd.DataFrame(data=j_rawdata_ad)
j_rawdata_k8s = pd.DataFrame(data=j_rawdata_k8s)
j_rawdata_hyscal = pd.DataFrame(data=j_rawdata_hyscal)

j_rawdata.to_csv('dataset/elapsed.csv')
j_rawdata_ad.to_csv('dataset/elapsed_ad.csv')
j_rawdata_k8s.to_csv('dataset/elapsed_k8s.csv')
j_rawdata_hyscal.to_csv('dataset/elpased_hyscal.csv')
