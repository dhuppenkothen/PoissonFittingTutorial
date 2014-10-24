
# coding: utf-8

# ## A quick Poisson fitting tutorial in python
# 
# Requires:
# - numpy
# - scipy
# - matplotlib
# - (emcee; if MCMC is something you're interested in)
# 
# 
# Data from the Chandra X-ray Satellite comes as images. These images are photon counting data, that is each pixel records an integer number of photons. 
# Data of this kind follows a Poisson distribution, that is, if there was no signal (i.e. only background noise), the resulting photons would follow a Poisson distribution with a single parameter (the background rate). This means that the value in each pixel $y$ is picked from the following distribution:
# 
# $$P(y|\lambda) = \frac{e^{-\lambda}\lambda^{y}}{y!}$$ ,
# 
# where $\lambda$ is the model parameter describing the expected count rate.
# 
# The function we are interested in is the *likelihood* of the data. The likelihood describes the probability of observing the data we've measured, conditioned on a *physical* model (your surface brightness model) and a *statistical* model (the Poisson distribution). The physical model describes what you expect your image to look like if there was no noise. The statistical model describes the uncertainty in the measurements because there is noise present. 
# The likelihood function is the product of the probabilities of all pixels, which is the probability of observing a number of counts in a pixel, $y_i$ under the assumption of a model count rate in that same pixel, $m_i$, multiplied together for all $N$ pixels.
# 
# All in all, the Poisson likelihood for a given physical model $m(\mathbf{\theta})$, which depends on a set of $K$ parameters $\mathbf{\theta} = \{\theta_1, \theta_2, ... , \theta_k\}$ looks like this:
# 
# $$L(\mathbf{\theta}) = P(\mathbf{y}|\mathbf{\theta}, H) = \prod_{i=0}^N{(\frac{e^{-m_i(\mathbf{\theta})}m_i(\mathbf{\theta})^{y_i}}{y_i!})}$$
# 
# The "best fit" parameters of your physical models are those that maximise the likelihood. In practical terms, the likelihood is often very large, and it is more convenient to maximise the logarithm of the likelihood. Because it is also easier for numerical reasons to minimise a function instead of maximising it, in practice you want to minimise the -log-likelihood, i.e. this quantity:
# 
# $$ 
# -\log{(L(\mathbf{\theta})} = \sum_{i=0}^{N}{(-m_i(\mathbf{\theta}) + y_i\log{(m_i(\mathbf{\theta}))} - \log{(y_i!)}    )}
# $$
# 
# You can separate out the various sums, and the logarithm of $y_i!$ becomes a $\Gamma$-function, which is in `scipy.special`. 
# 
# Below is some simple code for the Poisson likelihood of a given model and a stupidly simple example. Because I don't have image data, below I've used a lightcurve, but the whole thing should generalise easily to 2D and a more complicated model.
# 
# First, let's make some fake data. For now, this data is just a flat light curve of 1000 seconds duration and a time resolution of 1 second, with a background count rate of 10 counts/s. 

# In[33]:

import numpy as np
import scipy.stats
import scipy.optimize

## step 1: make some fake data, just a flat light curve with a 
## background parameter of 10

# time array
times = np.arange(0,1000,1)

# data array, pick from a Poisson distribution with mean rate=10
counts = np.random.poisson(10, size=len(times))


# Next, let's define the model for what the background should be. In our case, it's just a flat background with a single parameter that describes the background count rate (which, at this point, we pretend we don't know).

# In[34]:


## function that describes the model
## this could have more parameters if your model is more complex
def mymodel(times, rate_par):
    ## this can be as complex as you like
    model_counts = np.ones(len(times))*rate_par
    return model_counts
    


# Now we've got some data and we've got a physical model ("only background counts of an unknown flux") we'd like to use to describe this data. Now we need to define the Poisson likelihood, i.e. our statistical model.

# In[35]:

## this is the positive log-likelihood
## you can derive this easily from the definition of the poisson distribution
## rate is the 
def poisson_loglike(counts, model_counts):
    ## now we can compute the log-likelihood:
    llike = -np.sum(model_counts) + np.sum(counts*np.log(model_counts))-np.sum(scipy.special.gammaln(counts + 1))
    return llike


# Now we can put the two together and compute the thing we actually want to minimise. This is also often called the *deviance*, for unknown (to me) reasons:

# In[36]:

## model is the function that describes the physical model
## parameters is a list with parameters (one or more)
## times and counts are the data
def deviance(model, parameters, times, counts):
    model_counts = model(times, *parameters)
    return -2.*poisson_loglike(counts, model_counts)


# This you can now feed into a minimisation algorithm. But actually, 
# it's much easier and more convenient to put all of this stuff into a small simple class like this:

# In[37]:

logmin = 1000000000000.

class PoissonLikelihood(object):
    ### x,y = Data
    ### func = physical model
    def __init__(self, x, y, func):
        self.x = x
        self.y = y

        ### func is a parametric model
        self.func = func

        return
    
    def loglikelihood(self, t0):
        ## compute model counts
        model_counts = self.func(self.x, *t0)

        ## compute Poisson likelihood
        loglike = -np.sum(model_counts) + np.sum(self.y*np.log(model_counts))                    -np.sum(scipy.special.gammaln(self.y + 1))

        ## deal with NaN and inf values of the log-likelihood
        ## if either of these is true, choose a ridiculously small value
        ## to make the model *really* unlikely, but not infinite
        if np.isnan(loglike):
            loglike = -logmin
        elif loglike == np.inf:
            loglike = -logmin

        return loglike

    def __call__(self, t0, neg=False):
        lpost = self.loglikelihood(t0) 

        ## the negative switch makes sure you can call both
        ## the log-likelihood, and the negative log-likelihood
        if neg == True:
            return -lpost
        else:
            return lpost
    

    


# The class structure sets this problem up in a convenient way for fitting. Below, we're just plotting some guesses for the likelihood function to see how it changes with different values for the parameter. As you should be able to see, it gets larger (less negative) as we get closer to the true value of the rate (i.e. 10).

# In[38]:

## define PoissonLikelihood object:
pl = PoissonLikelihood(times, counts, mymodel)


## print the likelihood function for some guesses to see its behaviour:
guesses = [[5.], [7.], [10.], [15.], [20.]]
for g in guesses:
    print(pl(g, neg=False))



# Now let's do some actual fitting using scipy.optimize:

# In[39]:


## define your fit method of choice, look at documentation of scipy.optimize for details
## let's use the BFGS algorithm for now, it's pretty good and stable, but you can use others
## note that the output will change according to the algorithm, so check the documentation
## for what your preferred algorithm returns
fitmethod = scipy.optimize.fmin_bfgs


## set neg=True for negative log-likelihood

neg = True

initial_guess = [5.]
## do the actual fit
popt, fopt, gopt, covar, func_calls, grad_calls, warnflag  = fitmethod(pl, initial_guess, args=(neg,), full_output=True)

## popt are the optimum parameter values
## fopt is the likelihood function at optimum
## gopt is the value of the minimum of the gradient of the likelihood function
## covar is the inverse Hessian matrix, can be used for error calculation
## func_calls: number of function calls made
## grad_calls: number of calls to the gradient

print("Optimum parameter values: " + str(popt))
print("Likelihood at optimimum parameter values: " + str(fopt))


# In[40]:

if warnflag == 1:
    print("Maximum number of iterations exceeded")
elif warnflag == 2:
    print("Gradient and/or function calls not changing")
else:
    print("No clue")


# The returned value `bopt` is what we call the inverse of the Hessian matrix. The Hessian matrix is the matrix of second derivatives of the likelihood function with respect to each parameter in the model. This is computed numerically by the algorithm doing the fitting. 
# One can show that the inverse of this matrix, called the *covariance matrix*, describes the variances and covariances between parameters (i.e. how much they vary with themselves, and with each other parameter). This tells you something about the uncertainty in the parameters (via the variance) and how much they correlate  with each other (the covariances).
# You can compute the standard errors on your parameters by taking the square-root of the diagonal of this matrix (the square root of the variances):
# 

# In[41]:

stderr = np.sqrt(np.diag(covar))
print("standard error on the parameters: " + str(stderr))


# In our stupid simple example, there's only one parameter, so only one element in the inverse Hessian matrix. If you have two parameters, this matrix will have four elements, for three parameters 9 elements and so forth.
# 
# I'm still working on what the right statistic would be for testing the goodness-of-fit. I will add this when I've figured out what the most appropriate choice would be.
# 
# 
# ## MCMC for this problem using *emcee*
# 
# Markov Chain Monte Carlo is a powerful technique to recover complex probability distributions. For example, imagine your negative log-likelihood function has two minima for a given parameter. In this case, the fitting algorithm may end up at one minimum or another, depending on the starting value, and you will never know about the other. Or, imagine that your errors are skewed: your estimate may be much more uncertain in one direction than another. In this case, your standard errors will not reflect reality very well.
# In this case, it can be useful to *sample* the probability distribution instead of fit it in order to retain enough information about the distribution to make useful estimates. 
# 
# This is not an intro into MCMC, but there are many good tutorials out there. If you'd like to learn about MCMC, I'd recommend to write your own Metropolis-Hastings sampler. 
# 
# Below, I'll be using an out-of-the-box, very stable and well-written code called *emcee* to run MCMC on the likelihood above. This may take a while to run, depending on the number of data points and the complexity (number of parameters) of your model

# In[42]:

import emcee

## number of walkers in emcee
## should be large (>100)
nwalkers = 500

### number of dimensions for the Gaussian seeds (= number of parameters)
ndim = len(popt)

### sample random starting positions for each of the walkers
p0 = [np.random.multivariate_normal(popt,covar) for i in xrange(nwalkers)]

### initialize sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, pl, args=[False])

### run burn-in phase and reset sampler
burnin_samples = 200
pos, prob, state = sampler.run_mcmc(p0, burnin_samples)
sampler.reset()


### run actual MCMCs
niter = 100
sampler.run_mcmc(pos, niter, rstate0=state)

### list of all samples stored in flatchain
mcall = sampler.flatchain

### print meanacceptance rate for all walkers and autocorrelation times
print("The ensemble acceptance rate is: " + str(np.mean(sampler.acceptance_fraction)))
L = np.mean(sampler.acceptance_fraction)*len(mcall)
acceptance = np.mean(sampler.acceptance_fraction)
#try:
#    self.acor = sampler.acor
#    print("The autocorrelation times are: " +  str(sampler.acor))
#except ImportError:
#    print("You can install acor: http://github.com/dfm/acor")
#    self.acor = None
#except RuntimeError:
#    print("D was negative. No clue why that's the case! Not computing autocorrelation time ...")
#    self.acor = None
#except:
#    print("Autocorrelation time calculation failed due to an unknown error: " + sys.exc_info()[0] + ". Not computing autocorrelation time.")
#    self.acor = None



# `mcall` is a list of parameter values. You can make a histogram of it to see the distribution of your parameter(s).

# In[43]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt

plt.hist(mcall[:,0], bins=30);


# For multi-parameter problems, you can make a matrix plot that plots scatter plots for each parameter against each other parameter, and histograms for each parameter individually. This is a tremendously useful diagnostic tool in determining whether your distributions are skewed, or whether there are significant correlations between parameters. 
# 
# Here's code for the matrix plot. It makes a simple scatter plot like the one above in the case of a one-parameter model, and a matrix plot for a model with several parameters.

# In[44]:


## function that makes the scatter plot
def scatter_plot(samples, ndims):
    fig, axes = plt.subplots(ndims,ndims,figsize=(15,15))

    ### for one-parameter model, make scatter plot
    if ndims == 1:
        axes.hist(samples, bins=20)
        
    ### for more than one parameter, make matrix plot
    else:
        for i in xrange(ndims): ## x dimension
            for j in xrange(ndims): ## y dimension
                if i == j: 
                    axes[i,j].hist(samples[:,i], bins=20)
                else:
                    axes[i,j].scatter(samples[:,i], samples[:,j], color="black")

    return


## number of dimensions for the plot = number of parameters
ndims = mcall.shape[1]

## number of MCMC samples
nsamples = mcall.shape[0]

## make scatter plot for real model:
scatter_plot(mcall, ndims)




# In order to see the scatter plot properly, below I will make some random multi-dimensional samples. Imagine these are MCMC samples from a model with four parameters. I'm using a multi-variate Gaussian distribution for convenience.

# In[45]:


means = [1,2,6,4]
covariances = [[3,2,1,3],[1,2,1,4], [1,2,3,2], [3,1,3,1]]
    
rand_data = np.random.multivariate_normal(means, np.array(covariances), size=100)
ndims = rand_data.shape[1]

scatter_plot(rand_data, ndims)




# Note that the top and the bottom are mirrored: the scatter plot in the top row, second column is the same as the second row, first column plot (i.e. we're making scatter plots of the same parameters against each other), just mirrored. You can see that some parameters have strong correlations. The histograms all look quite Gaussian, but this is because I put in Gaussian distributions to simulate the samples.
# 
# The advantage of MCMC is that you can easily make simulations from the samples you drew from the model. This can be quite useful when you'd like to, for example, make sure the errors in your parameter estimation are properly accounted for in whatever final result you're interested in (I guess in your case, the cavity regions in the image). 
# The basic procedure works like this:
# - randomly pick a parameter set from your MCMC sample
# - make an image from that model
# - for each pixel in this image, pick from a Poisson distribution where the distribution's parameter $\lambda$ is the pixel value derived from your model. We're going to pretend that this is data (without cavity)
# - run whatever cavity detection procedure you use on this simulated image and compute whatever final result you're interested in
# - repeat these four steps many (say, 1000$ times
# - compare the numbers derived from your simulations to that from your real observed image. If the numbers are very similar, you probably don't have a cavity. If they are very different, either there's something real there or your background model is wrong.
# 
# If you're interested in this sort of stuff, let me know and we can talk about it in more detail. 

# In[ ]:



