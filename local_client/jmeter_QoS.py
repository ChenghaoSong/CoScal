import pandas as pd
import numpy as np

state_cur = []
for i in range(200, 220):
    jmeter = pd.read_csv('f:/jmeter_rawdata/jmeter_rawdata_%s.csv' % i)
    elapsed = np.array(jmeter['elapsed'])
    timeStamp = jmeter['timeStamp']
    success = np.array(jmeter['success'].astype('int'))
    latency = np.array(jmeter['Latency'])
    connect = np.array(jmeter['Connect'])
    lenth = len(timeStamp)
    timeStamp_len = timeStamp[lenth - 1] - timeStamp[0]
    state_cur_val = (np.mean(elapsed) * 0.05 + np.mean(latency) * 0.1 + np.mean(connect) * 0.05 + (
                1 - np.mean(success)) * 2 + (timeStamp_len-300000)/300000)/8
    if state_cur_val > 4:
        state_cur_val = 4
    state_cur.append(round(state_cur_val))
    csv = pd.DataFrame(data=state_cur)
    names = ['state']
    csv.columns = names
    csv.to_csv('/data/state.csv', mode='w')