# -*- coding: utf-8 -*-
import json
import os
from time import sleep
import pandas as pd

tps_csv = []
for i in range(1, 100):
    k = int(i * 50 * 60)
    os.system("jmeter -n -t nmontest.jmx -l loadbench/jmeter_rawdata_%s.jtl -JthreadNum=%s -e -o /home/sch/jmeter_client/loadbench/Report" % (i, k))
    sleep(3)
    dataJson = json.load(open('/home/sch/jmeter_client/loadbench/Report/statistics.json', encoding='UTF-8'))
    data = dataJson["Total"]
    tps = round(data["throughput"])
    err = round(data["errorPct"])
    if err > 30:
        break
    tps_csv.append(tps)
    csv = pd.DataFrame(data=tps_csv)
    names = ['tps', 'requests']
    csv.columns = names
    csv.to_csv('/home/sch/jmeter_client/loadtest/tps.csv', mode='w')
    os.system("rm -rf /home/sch/jmeter_client/loadtest/Report")
