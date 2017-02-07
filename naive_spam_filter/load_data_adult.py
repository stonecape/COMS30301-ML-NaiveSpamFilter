#!/usr/bin/env python

import numpy as np
from train import trainMarkedWords,classLables
from scipy.special import expit as sigmoid

# predictions from a random forest
#input_file = 'adult/y_and_p.csv'

print("loading data...")

#y_and_p = np.loadtxt( input_file, delimiter = ',' )
#
#y = y_and_p[:,0]
#p = y_and_p[:,1]
print(len(trainMarkedWords))
print(len(classLables))

y = np.array(classLables)
p = np.array(trainMarkedWords)
#print(p[1])
# y need to be 0/1
y[y == -1] = 0

