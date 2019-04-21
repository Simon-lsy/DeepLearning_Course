import numpy as np
import keras
from keras import backend as K
from matplotlib import pyplot as plt
import pickle

pkl_file1 = open('data1.pickle', 'rb')

data1 = pickle.load(pkl_file1)

pkl_file2 = open('data2.pickle', 'rb')

data2 = pickle.load(pkl_file2)

pkl_file3 = open('data3.pickle', 'rb')

data3 = pickle.load(pkl_file3)


plt.plot(range(30), data1['acc'], label='acc1')
# plt.plot(range(30), data1['val_acc'], label='val_acc1')
plt.plot(range(30), data2['acc'], label='acc2')
# plt.plot(range(30), data2['val_acc'], label='val_acc2')
plt.plot(range(30), data3['acc'], label='acc3')
# plt.plot(range(30), data3['val_acc'], label='val_acc3')
plt.legend()
plt.show()