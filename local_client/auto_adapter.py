# -*- coding: utf-8 -*-
from time import sleep
import pandas as pd
import numpy as np
import paramiko

sleep(60)
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
jmeter_min = pd.read_csv('jmeter_rawdata_adjust/jmeter_rawdata_%s.csv' % i)
elapsed_min = np.array(jmeter_min['elapsed'])
timeStamp_min = jmeter_min['timeStamp']
success_min = np.array(jmeter_min['success'].astype('int'))
latency_min = np.array(jmeter_min['Latency'])
connect_min = np.array(jmeter_min['Connect'])
lenth_min = len(timeStamp_min)
timeStamp_len_min = (timeStamp_min[lenth_min-1] - timeStamp_min[0]) * 5
# QoS: version 3
state_cur_val_min = (np.mean(elapsed_min) * 0.05 + np.mean(latency_min) * 0.1 + np.mean(connect_min) * 0.05 + (
                1 - np.mean(success_min)) * 2 + (timeStamp_len_min-300000)/300000)/8
if state_cur_val_min > 4:
    state_cur_val_min = 4
state_now = int(round(state_cur_val_min))
ssh.connect(hostname='172.20.110.72', port=22, username='root', password='1212')
if state_curr < state_now:
    ssh.exec_command('docker update --cpus 8 -m 3.5g 946f')
    ssh.exec_command('docker update --cpus 8 -m 3.5g 5ec4')
    ssh.close()
elif state_curr == state_now:
    sleep(10)
    ssh.close()
elif state_curr > state_now:
    ssh.exec_command('docker update --cpus 8 -m 3.752g 946f')
    ssh.exec_command('docker update --cpus 8 -m 3.752g 5ec4')
    ssh.close()
