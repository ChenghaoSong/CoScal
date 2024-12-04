# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

j_hor = []
j_ver = []
j_bro = []
j_non = []

for i in range(0, 20):
    hor = pd.read_csv('lowtest_hor/hor_%s.csv' % i)
    ver = pd.read_csv('lowtest_ver/ver_%s.csv' % i)
    bro = pd.read_csv('lowtest_brownout/brownout_%s.csv' % i)
    non = pd.read_csv('lowtest_raw/jmeter_rawdata_%s.csv' % i)
    hor = hor['success'].astype('int')
    ver = ver['success'].astype('int')
    bro = bro['success'].astype('int')
    non = non['success'].astype('int')

    hor = np.mean(hor)
    ver = np.mean(ver)
    bro = np.mean(bro)
    non = np.mean(non)
    j_hor.append(hor)
    j_ver.append(ver)
    j_bro.append(bro)
    j_non.append(non)

hor_save = pd.DataFrame(j_hor)
ver_save = pd.DataFrame(j_ver)
bro_save = pd.DataFrame(j_bro)
hor_save.to_csv('csv/success_hor.csv', header=False)
ver_save.to_csv('csv/success_ver.csv', header=False)
bro_save.to_csv('csv/success_bro.csv', header=False)

plt.xlabel('round')
plt.ylabel('success')
plt.plot(j_hor, label='hor', color='red')
plt.plot(j_ver, label='ver', color='blue')
plt.plot(j_bro, label='brownout', color='green')
# plt.plot(j_non, label='raw', color='coral')
plt.legend()
plt.savefig('img/success_compare.png')
plt.show()
