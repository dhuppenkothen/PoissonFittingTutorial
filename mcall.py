#!usr/bin/env python

import pickle
import emcee
import numpy
from matplotlib import pyplot

file = open("positiveLLmcall")
sampler = pickle.load(file)

mcall = sampler#.flatchain

fig = pyplot.figure()
ax = fig.add_subplot(1,1,1)

ax.hist(mcall[:,1], bins=100)
print len(mcall[:,1])
#ax.scatter(mcall[:,1], mcall[:,2])
pyplot.show()


pyplot.show()

