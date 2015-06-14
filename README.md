Bayesian simplicial clustering
==============================

The Bayesian simplicial mixture (BSM) model implements simulation and inference algorithms for a simplicial mixture model with component multi-variate *t* distributions.

Simplicial models are typically used in text-based information retrieval. For instance, the latent Dirichlet allocation (LDA) model allocates topics to documents within a corpus based on their word statistics. In the Bayesian simplicial mixture (BSM) model, documents are replaced by sets of continuous data. Each set is assumed to originate from a simplicial mixture of classes with set-specific mixing proportions. Words in the dictionary are replaced by symbols, where each symbol is a mixture component, and classes arise as mixtures of symbols.

The BSM models each component as a heavy-tailed distribution with symbol-specific parameters. Parameters are equipped with conjugate prior distributions as per the Bayesian paradigm. The posterior distributions over both parameters and hidden variables are estimated by an approximate variational inference algorithm.
