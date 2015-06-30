
import math,numpy,pylab

from matplotlib import patches,pyplot
from numpy import linalg
from scipy.cluster import vq

from mixmod import model

def scatterplot(obs,assign,loc=None,scale=None,colormap='jet',numcolor=16):

    numdim,numpoint=numpy.shape(obs)

    # Create a figure and a matrix of axis pairs.
    fig,axis=pyplot.subplots(numdim,numdim)

    # Adjust the axes and the tick marks.
    fig.subplots_adjust(hspace=0,wspace=0)
    for h in axis.flat:
        h.xaxis.set_visible(False)
        h.yaxis.set_visible(False)
        if h.is_first_row():
            h.xaxis.set_ticks_position('top')
            h.xaxis.set_visible(True)
        if h.is_last_row():
            h.xaxis.set_ticks_position('bottom')
            h.xaxis.set_visible(True)
        if h.is_first_col():
            h.yaxis.set_ticks_position('left')
            h.yaxis.set_visible(True)
        if h.is_last_col():
            h.yaxis.set_ticks_position('right')
            h.yaxis.set_visible(True)

    colormap=pyplot.get_cmap(colormap)

    if numpy.ndim(assign)>1:

        numcateg,numpoint=numpy.shape(assign)

        # Create a palette by quantizing a set of weighted base colors.
        color={k:colormap(float(k)/float(numcateg)) for k in range(numcateg)}
        palette,ind=vq.kmeans2(numpy.dot(assign.transpose(),list(color.values())),
                               min(numcolor,numpoint),minit='points')

        palette={k:numpy.clip(palette[k,:],0.0,1.0) for k in numpy.unique(ind)}

    else:

        numcateg=max(assign)+1

        # Store a base color for each category.
        color={k:colormap(float(k)/float(numcateg)) for k in range(numcateg)}

        ind=assign
        palette=color

    categ={k:[] for k in palette.keys()}

    # Group the data into categories
    # according to their palette indices.
    for i,k in enumerate(ind):
        categ[k].append(i)

    # Populate the plots.
    for i in range(numdim):
        for j in range(numdim):
            if i!=j:

                # Plot one dimension of the
                # data against another, with
                # colors from the palette.
                for k,ind in categ.items():
                    axis[i,j].scatter(obs[i,ind],obs[j,ind],
                                      color=palette[k],
                                      marker='.')

                if loc is not None and scale is not None:
                    for k in color.keys():

                        # Decompose the corresponding sub-matrix of the scale matrix.
                        eigval,eigvec=linalg.eigh(scale[k][numpy.ix_([i,j],[i,j])])

                        width,height=numpy.sqrt(eigval)
                        angle=numpy.degrees(numpy.arctan2(*eigvec[:,0][::-1]))

                        # Create an ellipse depicting the sub-matrix.
                        ellip=patches.Ellipse(xy=loc[k][numpy.ix_([i,j])],
                                              width=3.0*width,
                                              height=3.0*height,
                                              angle=angle,
                                              facecolor='none',
                                              edgecolor=color[k],
                                              linewidth=2,
                                              zorder=100)

                        axis[i,j].add_artist(ellip)

            else:
                if numpy.ndim(assign)>1:

                    # Create a weighted histogram with the base colors.
                    axis[i,j].hist([obs[i,:] for k in range(numcateg)],
                                   weights=[assign[k,:] for k in range(numcateg)],
                                   color=color.values(),
                                   histtype='bar',
                                   edgecolor='none')

                else:

                    # Create a histogram with the base colors.
                    axis[i,j].hist([obs[i,ind] for ind in categ.values()],
                                   color=color.values(),
                                   histtype='bar',
                                   edgecolor='none')

    return fig,axis

def likplot(bound,color='blue'):

    # Create a figure
    # and a pair of axes.
    fig=pyplot.figure()
    axis=fig.add_subplot(111)

    # Plot the lower bound on the marginal log-likelihood of the data.
    axis.plot(bound,color=color,marker='.',linewidth=2,markersize=10)

    axis.set_xlabel('Number of iterations')
    axis.set_ylabel('Lower bound on the\nmarginal log-likelihood of the data')

    return fig,axis

# Set the size
# of the problem.
numgroup=2
numcomp=3
numdim=5
numsamp=10
numpoint=20

param={'prop':5.0,
       'loc':0.1,
       'disp':10.0,
       'weight':3.0}

mod=model(numgroup,numcomp,numdim)

# Set the hyper-parameters.
for i in range(numgroup):
    mod.groups[i].alpha=param['prop']
for i in range(numcomp):
    mod.components[i].omega=param['loc']
    mod.components[i].eta=max(param['disp'],numdim)

# Generate a collection of sets of complete data.
groups,components,weight,obs=mod.simulate(*[numpoint for i in range(numsamp)],
                              alpha=param['prop'],nu=param['weight'])

# Create a matrix of scatter plots of the data and color each observation
# according to the mixture component responsible for generating it.
fig,axis=scatterplot(numpy.concatenate(obs,axis=1),numpy.concatenate(components))

fig.canvas.set_window_title('Observations')

# Infer the approximate posterior probabilities and weights.
prob,weight,bound=mod.infer(*obs,alpha=param['prop'],nu=param['weight'])

loc,scale=zip(*[(mod.components[k].mu,mod.components[k].sigma) for k in range(numcomp)])

# Create another matrix of scatter plots, but
# this time color the points according to their
# probabilistic component assignments.
fig,axis=scatterplot(numpy.concatenate(obs,axis=1),
                     numpy.concatenate([p.sum(axis=0) for p in prob],axis=1),
                     loc=loc,scale=scale)

fig.canvas.set_window_title('Clustering results')

# Plot the variational lower bound
# on the marginal log-likelihood of
# the data after each iteration.
fig,axis=likplot(bound)

fig.canvas.set_window_title('Variational lower bound')

pyplot.show()
