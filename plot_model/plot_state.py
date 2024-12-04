# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('state.csv')

state = dataset['state']

plt.xlabel('round')
plt.ylabel('state')

plt.plot(state, label='raw', color='coral')
plt.legend()
plt.savefig('plot_state.png')
plt.show()
