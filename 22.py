# coding: utf-8
import numpy as np
from rnnmath import *
from rnn1 import *

U = np.array([(0.5,0.3), (0.4,0.2)])
V = np.array([(0.2,0.5,0.1), (0.6,0.1,0.8)])
W = np.array([(0.4,0.2),(0.3,0.1),(0.1,0.7)])
s0 = np.array([(0.3,0.6)])
x = np.array([(0,1,0)])
print "111",(V.dot(x.T)+U.dot(s0.T))
question3=sigmoid((V.dot(x.T)+U.dot(s0.T))).T
print 'zhongtu',W.dot(question3.T)
question4=softmax(W.dot(question3.T)).T
print "aaa", -np.log(0.366)
print 'ooo', -np.log(0.603)
print "aaaa",-np.log(0.317)
#
# print V,U,W,s0,x
print question3
print question4

# RNNN=RNN(1,2)
# diccc = {}
# diccc['111'] = 123
# diccc['gdg'] = 4343
# print diccc
#
# pen = open('resulttt.txt', 'a')
# for g in diccc.keys():
#     pen.write(g)
#     pen.write(str(diccc[g]))
#     pen.write('\n')
#
#
#     #     print xx
# #     print yy
# #     for b in xx:
# #         print b
# # # a=np.array([(2,1,3),(2,3,2)])
# # # testtt([x,x2,x3],[d,d2,d3])
# # RNNN.compute_loss([x,x2,x3],[d,d2,d3])
#
# # 0.5 0.3
# # 0.4 0.2
# #  [0 1 0]
# # T
# # V =
# #
# # 0.2
# # 0.5
# # 0.1
# # 0.6
# # 0.1
# # 0.8
# # W =
# # 
# # 
# # 0.4 0.2
# # 0.3 0.1
# # 0.1 0.7
# # 
# # 