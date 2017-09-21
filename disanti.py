from rnn import *
import numpy as np
from utils import *
import re

# a=load_dataset('data/vocab.ptb.txt')
# print a[0][10000]
# b=load_dataset('data/vocab.ptb.txt')
# bb=0.
# for s in b[0]:
#     c = re.split('\t',s[1])
#     # print c[1]
#     bb+=float(c[1])
a = 'RNN_50_LB_5_LR_0.05'
f = re.split('_',a)
print f