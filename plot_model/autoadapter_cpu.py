# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(
    '/data/process_monitor_python3_929159.csv', nrows=1200)
cpu = dataset['cpu%']
# 20 cores
cpu = cpu / 20
dataset['cpu%'] = cpu
mem = dataset['mem%']
cpu_add = 0
mem_add = 0
lenth = len(cpu)
for i in range(0, lenth):
    cpu_add = cpu_add + cpu[i]
    mem_add = mem_add + mem[i]
cpu_avg = cpu_add / lenth
mem_avg = mem_add / lenth

print(cpu_avg)
print(mem_avg)
dataset.to_csv('/csv/autoadapter_cpu_mem.csv')
# plt.xlabel('cpu/memory')
# plt.ylabel('elapsed')
plt.plot(cpu, label='cpu', color='red')
plt.plot(mem, label='mem', color='blue')
plt.legend()
plt.show()
