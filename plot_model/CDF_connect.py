# -*- coding: utf-8 -*-
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd

dataset_connect = pd.read_csv(
    '/data/connect.csv')
sample = dataset_connect['0']
ecdf = sm.distributions.ECDF(sample)
x = np.linspace(0, max(sample))
y = ecdf(x)
plt.plot(x, y, linewidth='1', label='DoScal')

dataset_connect_ad = pd.read_csv(
    '/data/connect_ad.csv')
sample = dataset_connect_ad['0']
ecdf = sm.distributions.ECDF(sample)
x = np.linspace(0, max(sample))
y = ecdf(x)
plt.plot(x, y, linewidth='1', label='CoScal')

dataset_connect_ad = pd.read_csv(
    '/data/connect_k8s.csv')
sample = dataset_connect_ad['0']
ecdf = sm.distributions.ECDF(sample)
x = np.linspace(0, max(sample))
y = ecdf(x)
plt.plot(x, y, linewidth='1', label='KuScal')

dataset_connect_ad = pd.read_csv(
    '/data/connect_hyscal.csv')
sample = dataset_connect_ad['0']
ecdf = sm.distributions.ECDF(sample)
x = np.linspace(0, max(sample))
y = ecdf(x)
plt.plot(x, y, linewidth='1', label='HyScal')
# x = np.linspace(min(sample), max(sample))
plt.xlabel('connect', fontdict={'weight': 'normal', 'size': 16})
plt.ylabel('CDF', fontdict={'weight': 'normal', 'size': 16})
plt.legend()
plt.grid(linestyle='--')
# plt.axis([0.9, 1, 0, 1])
plt.savefig('/plot_model/connect_cdf.png')
plt.show()
