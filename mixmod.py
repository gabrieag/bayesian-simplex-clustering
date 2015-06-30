#!/usr/bin/python

# Python implementation of a Bayesian simplicial mixture of
# multi-variate t distributions. The class implements methods
# for generating simulated data and estimating the parameters
# of the model.
# 
# Simplicial mixture models are typically used in text-based
# information retrieval, e.g. latent Dirichlet allocation (LDA).
# The LDA model allocates topics to a set of documents within a
# corpus based on their word statistics. Here, the documents are
# replaced by continuous data. Each set of data originates from
# a simplicial mixture of multi-variate t distributions with set-
# specific mixing proportions. An additional layer of latent
# variables interface the documents' topics and words.

import copy

import numpy as np
from numpy import linalg
from numpy import random

# Import the module-specific classes and functions.
from __dist__ import dirich, gaussgamma, gausswish
from __util__ import isconv, unique

class model():

    # Define a structure-like container
    # class for storing the distributions
    # over the model parameters.
    class paramdist:
        group = None
        comp = None

    def __init__(self, numgroup, numcomp, numdim, diag = False):

        # Check the size of the model.
        assert numgroup > 0 and numcomp > 0 and numdim > 0

        self.__size__ = numgroup, numcomp, numdim
        self.__prior__ = model.paramdist()

        dist = gaussgamma if diag else gausswish

        # Initialize the prior distributions over the model parameters.
        self.__prior__.group = [dirich(numcomp) for i in range(numgroup)]
        self.__prior__.comp = [dist(numdim) for i in range(numcomp)]

        self.__post__=None

    @property
    def group(self):

        # By default, select the posterior distributions over the model
        # parameters. If they are not initialized, then select the prior.
        dist = self.__post__ if self.__post__ is not None else self.__prior__

        return dist.group

    @group.setter
    def group(self, *group):

        numgroup, numcomp, numdim = self.__size__

        # Check that the number of
        # arguments is consistent with
        # the size of the model.
        assert len(group) == numgroup

        # Check that the arguments are Dirichlet distributions.
        assert all(isinstance(d, dirich) for d in group)

        # Set these as the prior distributions
        # over the group-specific parameters.
        self.__prior__.group = group

        self.__post__ = None

    @property
    def comp(self):

        # By default, select the posterior distributions over the model
        # parameters. If they are not initialized, then select the prior.
        dist = self.__post__ if self.__post__ is not None else self.__prior__

        return dist.comp

    @comp.setter
    def comp(self, *comp):

        numgroup, numcomp, numdim = self.__size__

        # Check that the number of
        # arguments is consistent with
        # the size of the model.
        assert len(comp) == numcomp

        # Check that the arguments are either Gauss-Gamma or Gauss-Wishart distributions.
        assert all(isinstance(d, gaussgamma) or isinstance(d, gausswish) for d in comp)

        # Set these as the prior distributions
        # over the component-specific parameters.
        self.__prior__.comp = comp

        self.__post__ = None

    def sim(self, *size, alpha = np.inf, nu = np.inf):

        # Check that the sizes and hyper-parameters are valid.
        assert all(n > 0 for n in size) and alpha > 0.0 and nu > 0.0

        numgroup, numcomp, numdim = self.__size__

        # By default, select the posterior distributions over the model
        # parameters. If they are not initialized, then select the prior.
        dist = self.__post__ if self.__post__ is not None else self.__prior__

        # Create a distribution over
        # the sample-specific parameters.
        prop = dirich(numgroup, alpha = alpha)

        # Generate the model-specific parameters.
        emiss = [p.rand() for p in dist.group]
        loc, disp = zip(*[p.rand() for p in dist.comp])

        group, comp, weight, obs = [], [], [], []

        for i, numpoint in enumerate(size):

            # Generate the group indices.
            group.append(prop.rand().cumsum().searchsorted(random.rand(numpoint)))

            comp.append(np.zeros(numpoint, dtype=int))
            weight.append(np.zeros(numpoint))
            obs.append(np.zeros([numdim, numpoint]))

            # Generate the component indices.
            for j, ind in unique(group[i]):
                comp[i][ind] = emiss[j].cumsum().searchsorted(random.rand(len(ind)))

            # Generate the observation weights.
            if np.isfinite(nu):
                weight[i] = random.gamma(nu/2.0, size = numpoint)/(nu/2.0)
            else:
                weight[i][:] = 1.0

            # Generate the observations.
            for j, ind in unique(comp[i]):
                scale = np.sqrt(weight[i][ind])
                obs[i][:, ind] = loc[j][:, np.newaxis] + np.dot(linalg.cholesky(disp[j]), 
                    random.randn(numdim, len(ind)))/scale[np.newaxis, :]

        return group, comp, weight, obs

    def infer(self, *obs, alpha = np.inf, nu = np.inf, initpost = True, 
        numiter = [10, 1000], noisetemp = 1.0e-2, reltol = 1.0e-6):

        numgroup, numcomp, numdim = self.__size__

        # Check that there the arguments are consistent with the size of the model.
        assert all(np.ndim(x) == 2 and d == numdim for x in obs for d, n in (x.shape,))

        numpoint = [n for x in obs for d, n in (x.shape,)]

        prior = self.__prior__
        post = self.__post__

        numsamp = len(obs)

        if post is None:

            post = model.paramdist()

            # Initialize the posterior distributions
            # over the model-specific parameters.
            post.group = copy.deepcopy(prior.group)
            post.comp = copy.deepcopy(prior.comp)

        # Initialize the distributions over the sample-specific parameters.
        prior.samp = [dirich(numgroup, alpha = alpha) for i in range(numsamp)]
        post.samp = [dirich(numgroup, alpha = alpha) for i in range(numsamp)]

        if initpost:

            # Initialize the distributions over
            # the sample-specific parameters.
            for i in range(numsamp):
                post.samp[i].alpha += numpoint[i]

            a = float(sum(numpoint))/float(numgroup)
            b = float(sum(numpoint))/float(numcomp)

            # Initialize the distributions over
            # the model-specific parameters.
            for i in range(numgroup):
                post.group[i].alpha += a
            for i in range(numcomp):
                post.comp[i].omega += b
                post.comp[i].eta += b

        prob = [None]*numsamp
        weight = [None]*numsamp

        bound = []

        for i in range(max(numiter)):

            bound.append(0.0)

            for j in range(numsamp):

                loglik = np.zeros([numcomp, numpoint[j]])

                if weight[j] is None:
                    weight[j] = np.zeros([numcomp, numpoint[j]])

                # Evaluate the expected log-likelihood
                # of the observations, and the expected
                # value of the weights.
                for k in range(numcomp):
                    loglik[k, :], weight[j][k, :] = post.comp[k].loglik(obs[j], nu = nu)

                # Compute the joint log-probabilities.
                prob[j] = post.samp[j].loglik().reshape([numgroup, 1, 1]) \
                    + np.reshape([q.loglik() for q in post.group], [numgroup, numcomp, 1]) \
                    + loglik[np.newaxis, :, :]

                logconst = prob[j].max(axis = 0).max(axis = 0)
                logconst += np.log(np.exp(prob[j] - logconst[np.newaxis, np.newaxis, :]) 
                    .sum(axis = 0).sum(axis = 0))

                # Normalize to obtain the probabilities.
                prob[j] = np.exp(prob[j] - logconst[np.newaxis, np.newaxis, :])

                if i == 0:

                    # Add a bit of noise in order to break ties.
                    prob[j] *= 1.0 - noisetemp*random.rand(numgroup, numcomp, numpoint[j])
                    prob[j] /= prob[j].sum(axis = 0).sum(axis = 0).reshape([1, 1, numpoint[j]])

                    prob[j][np.logical_or(np.isnan(prob[j]), np.isinf(prob[j]))] = \
                        1.0/(numgroup*numcomp)

                # Accumulate the log-normalization constants.
                bound[i] += logconst.sum()

            # Evaluate the lower bound on the marginal log-likelihood of the data.
            bound[i] -= sum(q.div(p) for p, q in zip(prior.samp, post.samp))\
                + sum(q.div(p) for p, q in zip(prior.group, post.group))\
                + sum(q.div(p) for p, q in zip(prior.comp, post.comp))

            for j in range(numsamp):

                # Accumulate the expected sufficient statistics.
                stat = post.samp[j].stat([prob[j].sum(axis = 1)])

                # Update the posterior distributions
                # over the sample-specific parameters.
                post.samp[j].copy(prior.samp[j]).update(stat)

            for j in range(numgroup):

                # Accumulate the expected sufficient statistics.
                stat = post.group[j].stat(p[j, :, :] for p in prob)

                # Update the posterior distributions
                # over the model-specific group parameters.
                post.group[j].copy(prior.group[j]).update(stat)

            scale = [p.sum(axis = 0) for p in prob]

            for j in range(numcomp):

                # Accumulate the expected sufficient statistics.
                stat = post.comp[j].stat(([x, w[j, :], s[j, :]] for x, w, s in 
                    zip(obs, weight, scale)), weighted = True, scaled = True)

                # Update the posterior distributions over
                # the model-specific component parameters.
                post.comp[j].copy(prior.comp[j]).update(stat)

            if i > min(numiter) and isconv(reltol, bound[1:i]):
                break

        self.__post__ = post

        return prob, weight, bound[:i]
