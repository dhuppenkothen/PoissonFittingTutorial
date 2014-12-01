#!usr/bin/env python

import pyfits
import numpy
from matplotlib import pyplot
import scipy
from scipy import special
import emcee
import pickle

data = numpy.loadtxt("Ellipsefitparams.txt")
fitparams = data[0,:]
covar = data[1:,:]

#print fitparams, covar


hdu_in_list = pyfits.open("2a0335_core_flux.fits")
hdu_in = hdu_in_list[0]
sb_data  = hdu_in.data
in_header = hdu_in.header

xdata = numpy.meshgrid(numpy.linspace(0,len(sb_data), num=len(sb_data)),numpy.linspace(0,len(sb_data),num=len(sb_data)))


def poisson(x, cp_x, cp_y, q, a, phi, logScool, rcool, betacool, logbackground):

    Scool = numpy.exp(logScool)
    background = numpy.exp(logbackground)

    theta = numpy.arctan2(x[1] - cp_x , x[0] - cp_y)
    radius = numpy.sqrt((x[0]-cp_x)**2 + ((x[1]-cp_y)/q)**2) * ( 1 + (a * numpy.cos(theta + phi)))
    profile = Scool*(1 + (radius/rcool) **2.)**(0.5 - 3*betacool) + background
    
    LL = (-numpy.sum(profile) + numpy.sum(sb_data*numpy.log(profile)) - numpy.sum(scipy.special.gammaln(sb_data+1)))

    return LL

a = poisson(xdata, *fitparams)

#number of walkers
nwalkers = 500

#number of dimensions (free params)
ndim = 9

### sample random starting positions for each of the walkers
p0 = [numpy.random.multivariate_normal(fitparams,covar) for i in xrange(nwalkers)]
print p0[:5]

### initialize sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, poisson, args=fitparams)


### run burn-in phase and reset sampler
print 'Burn-in phase..'
burnin_samples = 200
pos, prob, state = sampler.run_mcmc(p0, burnin_samples)
sampler.reset()

### run actual MCMCs
print 'Running MCMC'
niter = 100
sampler.run_mcmc(pos, niter, rstate0=state)

### list of all samples stored in flatchain
mcall = sampler.flatchain

file_name = "positiveLLmcall"
fileObject = open(file_name, 'wb')
pickle.dump(mcall, fileObject)
fileObject.close()

file_name2 = "positiveLLsampler"
fileObject2 = open(file_name2, 'wb')
pickle.dump(sampler,fileObject2)
fileObject2.close()

### print meanacceptance rate for all walkers and autocorrelation times
print("The ensemble acceptance rate is: " + str(numpy.mean(sampler.acceptance_fraction)))
L = numpy.mean(sampler.acceptance_fraction)*len(mcall)
acceptance = numpy.mean(sampler.acceptance_fraction)


def scatter_plot(samples, ndims):
    fig, axes = pyplot.subplots(ndims, ndims, figsize=(15,15))

    if ndims == 1:
        axes.hist(samples, bins=20)

    else:
        for i in xrange(ndims):
            for j in xrange(ndims):
                if i == j:
                    axes[i,j].hist(samples[:,i], bins=20)
                else:
                    axes[i,j].scatter(samples[:,i],samples[:,j], color="black")
    return

scatter_plot(mcall,ndim)

pyplot.show()
