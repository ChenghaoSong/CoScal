# -*- coding: utf-8 -*-
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd

dataset_success = pd.read_csv(
    '/data/success.csv')
sample = dataset_success['0']
ecdf = sm.distributions.ECDF(sample)
x = np.linspace(0, max(sample))
y = ecdf(x)
su = x
plt.plot(x, y, linewidth='1', label='success')

dataset_success_ad = pd.read_csv(
    '/data/success_ad.csv')
sample = dataset_success_ad['0']
ecdf = sm.distributions.ECDF(sample)
x = np.linspace(0, max(sample))
y = ecdf(x)
su_ad = x
plt.plot(x, y, linewidth='1', label='success_ad')

dataset_success_ad = pd.read_csv(
    '/data/success_k8s.csv')
sample = dataset_success_ad['0']
ecdf = sm.distributions.ECDF(sample)
x = np.linspace(0, max(sample))
y = ecdf(x)
su_k8s = x
plt.plot(x, y, linewidth='1', label='success_k8s')

dataset_success_ad = pd.read_csv(
    '/data/success_hyscal.csv')
sample = dataset_success_ad['0']
ecdf = sm.distributions.ECDF(sample)
x = np.linspace(0, max(sample))
y = ecdf(x)
su_hyscal = x
plt.plot(x, y, linewidth='1', label='success_hyscal')
# x = np.linspace(min(sample), max(sample))
plt.xlabel('success', fontdict={'weight': 'normal', 'size': 16})
plt.ylabel('CDF', fontdict={'weight': 'normal', 'size': 16})
plt.legend()
plt.grid(linestyle='--')
plt.axis([0.9, 1, 0, 1])
plt.savefig('/plot_model/success_cdf.png')
plt.show()
