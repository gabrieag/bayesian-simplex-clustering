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
from __dist__ import dirich as Dirichlet
from __dist__ import gaussgamma as GaussGamma
from __dist__ import gausswish as GaussInvWishart
from __util__ import isconv, unique

class model():

    # Define a structure-like container
    # class for storing the distributions
    # over the model parameters.
    class paramdist:
        groups = None
        components = None

    def __init__(self, num_groups, num_components, num_dimensions, 
        all_correlations = True):

        # The size of the model is determined by the number of groups, the
        # number of components, and the of dimensions in the data. All of these
        # must be strictly positive.
        assert num_groups > 0 and num_components > 0 and num_dimensions > 0

        self.__size = num_groups, num_components, num_dimensions
        self.__prior = model.paramdist()

        dist = GaussInvWishart if all_correlations else GaussGamma

        # Initialize the prior distributions over the model parameters.
        self.__prior.groups = [Dirichlet(num_components) for i in range(num_groups)]
        self.__prior.components = [dist(num_dimensions) for i in range(num_components)]

        self.__posterior = None

    @property
    def groups(self):

        # By default, select the posterior distributions over the model
        # parameters. If they are not initialized, then select the prior.
        dist = self.__posterior if self.__posterior is not None else self.__prior

        return dist.groups

    @groups.setter
    def groups(self, *groups):

        num_groups, num_components, num_dimensions = self.__size

        # Check that the number of
        # arguments is consistent with
        # the size of the model.
        assert len(groups) == num_groups

        # Check that the arguments are Dirichlet distributions.
        assert all(isinstance(group, Dirichlet) for group in groups)

        # Set these as the prior distributions
        # over the group-specific parameters.
        self.__prior.groups = groups

        self.__posterior = None

    @property
    def components(self):

        # By default, select the posterior distributions over the model
        # parameters. If they are not initialized, then select the prior.
        dist = self.__posterior if self.__posterior is not None else self.__prior

        return dist.components

    @components.setter
    def components(self, *components):

        num_groups, num_components, num_dimensions = self.__size

        # Check that the number of
        # arguments is consistent with
        # the size of the model.
        assert len(components) == num_components

        # Check that the arguments are either Gauss-Gamma or Gauss-Wishart distributions.
        assert all(isinstance(component, GaussGamma) or isinstance(component, GaussInvWishart) for component in components)

        # Set these as the prior distributions
        # over the component-specific parameters.
        self.__prior.components = components

        self.__posterior = None

    def simulate(self, *size, alpha = np.inf, nu = np.inf):

        # Check that the sizes and hyper-parameters are valid.
        assert all(n > 0 for n in size) and alpha > 0.0 and nu > 0.0

        num_groups, num_components, num_dimensions = self.__size

        # By default, select the posterior distributions over the model
        # parameters. If they are not initialized, then select the prior.
        dist = self.__posterior if self.__posterior is not None else self.__prior

        # Create a distribution over
        # the sample-specific parameters.
        prop_param = Dirichlet(num_groups, alpha = alpha)

        # Generate the model-specific parameters.
        emiss_param = [p.rand() for p in dist.groups]
        loc_param, disp_param = zip(*[p.rand() for p in dist.components])

        group_ind, comp_ind, obs_weight, obs = [], [], [], []

        for i, num_points in enumerate(size):

            # Generate the group indices.
            group_ind.append(prop_param.rand().cumsum().searchsorted(random.rand(num_points)))

            comp_ind.append(np.zeros(num_points, dtype=int))
            obs_weight.append(np.zeros(num_points))
            obs.append(np.zeros([num_dimensions, num_points]))

            # Generate the component indices.
            for j, ind in unique(group_ind[i]):
                comp_ind[i][ind] = emiss_param[j].cumsum().searchsorted(random.rand(len(ind)))

            # Generate the observation weights.
            if np.isfinite(nu):
                obs_weight[i] = random.gamma(nu/2.0, size = num_points)/(nu/2.0)
            else:
                obs_weight[i][:] = 1.0

            # Generate the observations.
            for j, ind in unique(comp_ind[i]):
                scale = np.sqrt(obs_weight[i][ind])
                obs[i][:, ind] = loc_param[j][:, np.newaxis] + np.dot(linalg.cholesky(disp_param[j]), 
                    random.randn(num_dimensions, len(ind)))/scale[np.newaxis, :]

        return group_ind, comp_ind, obs_weight, obs

    def infer(self, *observations, alpha = np.inf, nu = np.inf,
        init_posterior = True, num_iterations = [10, 1000],
        noise_level = 1.0e-2, rel_tolerance = 1.0e-6):

        num_groups, num_components, num_dimensions = self.__size

        # Check that there the arguments are consistent with the size of the model.
        assert all(np.ndim(x) == 2 and d == num_dimensions for x in observations for d, n in (x.shape,))

        num_points = [n for x in observations for d, n in (x.shape,)]

        prior = self.__prior
        posterior = self.__posterior

        num_samples = len(observations)

        if posterior is None:

            posterior = model.paramdist()

            # Initialize the posterior distributions
            # over the model-specific parameters.
            posterior.groups = copy.deepcopy(prior.groups)
            posterior.components = copy.deepcopy(prior.components)

        # Initialize the distributions over the sample-specific parameters.
        prior.samp = [Dirichlet(num_groups, alpha = alpha) for i in range(num_samples)]
        posterior.samp = [Dirichlet(num_groups, alpha = alpha) for i in range(num_samples)]

        if init_posterior:

            # Initialize the distributions over
            # the sample-specific parameters.
            for i in range(num_samples):
                posterior.samp[i].alpha += num_points[i]

            a = float(sum(num_points))/float(num_groups)
            b = float(sum(num_points))/float(num_components)

            # Initialize the distributions over
            # the model-specific parameters.
            for i in range(num_groups):
                posterior.groups[i].alpha += a
            for i in range(num_components):
                posterior.components[i].omega += b
                posterior.components[i].eta += b

        probabilities = [None]*num_samples
        weights = [None]*num_samples

        lower_bound = []

        for i in range(max(num_iterations)):

            lower_bound.append(0.0)

            for j in range(num_samples):

                log_likelihood = np.zeros([num_components, num_points[j]])

                if weights[j] is None:
                    weights[j] = np.zeros([num_components, num_points[j]])

                # Evaluate the expected log-likelihood
                # of the observations, and the expected
                # value of the weights.
                for k in range(num_components):
                    log_likelihood[k, :], weights[j][k, :] = posterior.components[k].loglik(observations[j], nu = nu)

                # Compute the joint log-probabilities.
                probabilities[j] = posterior.samp[j].loglik().reshape([num_groups, 1, 1]) \
                    + np.reshape([q.loglik() for q in posterior.groups], [num_groups, num_components, 1]) \
                    + log_likelihood[np.newaxis, :, :]

                norm_constant = probabilities[j].max(axis = 0).max(axis = 0)
                norm_constant += np.log(np.exp(probabilities[j] - norm_constant[np.newaxis, np.newaxis, :]) 
                    .sum(axis = 0).sum(axis = 0))

                # Normalize to obtain the probabilities.
                probabilities[j] = np.exp(probabilities[j] - norm_constant[np.newaxis, np.newaxis, :])

                if i == 0:

                    # Add a bit of noise in order to break ties.
                    probabilities[j] *= 1.0 - noise_level*random.rand(num_groups, num_components, num_points[j])
                    probabilities[j] /= probabilities[j].sum(axis = 0).sum(axis = 0).reshape([1, 1, num_points[j]])

                    probabilities[j][np.logical_or(np.isnan(probabilities[j]), np.isinf(probabilities[j]))] = \
                        1.0/(num_groups*num_components)

                # Accumulate the log-normalization constants.
                lower_bound[i] += norm_constant.sum()

            # Evaluate the lower bound on the marginal log-likelihood of the data.
            lower_bound[i] -= sum(q.div(p) for p, q in zip(prior.samp, posterior.samp))\
                + sum(q.div(p) for p, q in zip(prior.groups, posterior.groups))\
                + sum(q.div(p) for p, q in zip(prior.components, posterior.components))

            for j in range(num_samples):

                # Accumulate the expected sufficient statistics.
                statistics = posterior.samp[j].stat([probabilities[j].sum(axis = 1)])

                # Update the posterior distributions
                # over the sample-specific parameters.
                posterior.samp[j].copy(prior.samp[j]).update(statistics)

            for j in range(num_groups):

                # Accumulate the expected sufficient statistics.
                statistics = posterior.groups[j].stat(p[j, :, :] for p in probabilities)

                # Update the posterior distributions
                # over the model-specific group parameters.
                posterior.groups[j].copy(prior.groups[j]).update(statistics)

            scale = [p.sum(axis = 0) for p in probabilities]

            for j in range(num_components):

                # Accumulate the expected sufficient statistics.
                statistics = posterior.components[j].stat(([x, w[j, :], s[j, :]] for x, w, s in 
                    zip(observations, weights, scale)), weighted = True, scaled = True)

                # Update the posterior distributions over
                # the model-specific component parameters.
                posterior.components[j].copy(prior.components[j]).update(statistics)

            if i > min(num_iterations) and isconv(rel_tolerance, lower_bound[1:i]):
                break

        self.__posterior = posterior

        return probabilities, weights, lower_bound[:i]
