
import math,numpy

from numpy import linalg,random
from scipy import special

def isconv(tol,val):
    if len(val)<2:
        return False
    return max(2.0*abs(val[-1]-val[-2]),numpy.spacing(1.0-tol))<=tol*(abs(val[-1])+abs(val[-2]))

def unique(seq):
    order=numpy.argsort(seq)
    ind,=numpy.where(numpy.diff(seq[order])>0)
    ind=numpy.concatenate([numpy.array([0]),ind+1,numpy.array([numpy.size(seq)])])
    for i,j in zip(ind[:-1],ind[1:]):
        yield seq[order[i]],order[i:j]
