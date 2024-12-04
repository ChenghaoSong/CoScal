# -*- coding: utf-8 -*-
import json
import os
from time import sleep
import pandas as pd

tps_csv = []
for i in range(200, 300):
    os.system("jmeter -g /home/sch/jmeter_client/jmeter_rawdata_adjust/jmeter_rawdata_%s.csv -e -o /home/sch/jmeter_client/jmeter_rawdata_adjust/resultReport" % i)
    sleep(3)
    dataJson = json.load(open('/home/sch/jmeter_client/jmeter_rawdata_adjust/resultReport/statistics.json', encoding='UTF-8'))
    data = dataJson["Total"]
    tps = round(data["throughput"])
    tps_csv.append(tps)
    csv = pd.DataFrame(data=tps_csv)
    names = ['tps']
    csv.columns = names
    csv.to_csv('/home/sch/jmeter_client/jmeter_rawdata_adjust/tps.csv', mode='w')
    os.system("rm -rf /home/sch/jmeter_client/jmeter_rawdata_adjust/resultReport")
