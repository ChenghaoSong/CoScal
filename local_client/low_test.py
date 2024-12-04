# -- coding: utf-8 --
import os
from time import sleep
import pandas as pd
import numpy as np

switch_on = 0
state_cur = []
# pull requests via jmeter
for i in range(0, 20):
    if switch_on == 0:
        os.system("jmeter -n -t newplan.jmx -l lowtest_ver/ver_%s.csv -JthreadNum=1650000" % i)
    elif switch_on == 1:
        os.system("jmeter -n -t newplan.jmx -l lowtest_ver/ver_%s.csv -JthreadNum=825000" % i)
    jmeter = pd.read_csv('lowtest_ver/ver_%s.csv' % i)
    elapsed = np.array(jmeter['elapsed'])
    timeStamp = jmeter['timeStamp']
    success = np.array(jmeter['success'].astype('int'))
    latency = np.array(jmeter['Latency'])
    connect = np.array(jmeter['Connect'])
    lenth = len(timeStamp)
    timeStamp_len = timeStamp[lenth - 1] - timeStamp[0]
    # QoS: version 3
    state_cur_val = (np.mean(elapsed) * 0.05 + np.mean(latency) * 0.1 + np.mean(connect) * 0.05 + (
            1 - np.mean(success)) * 2 + (timeStamp_len - 300000) / 300000) / 8
    if state_cur_val > 4:
        state_cur_val = 4
    state_cur.append(int(round(state_cur_val)))
    csv = pd.DataFrame(data=state_cur)
    names = ['state']
    csv.columns = names
    csv.to_csv('lowtest_ver.csv', mode='w')
    # run requests_server.py
    if i > 4:
        with open('lowtest_hor.py', 'r') as f:
            exec(f.read())
    os.system("pkill -9 java")
