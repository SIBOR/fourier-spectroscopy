# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 05:29:55 2017

@author: jaymz
"""

from __future__ import division
import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.float
ctypedef np.float_t DTYPE_t

@cython.boundscheck(False)
def fourierTransC(np.ndarray[DTYPE_t,ndim=1] datX,np.ndarray[DTYPE_t,ndim=1] datY,np.float wi,np.float wf,np.int n):
    assert datX.dtype == DTYPE and datY.dtype == DTYPE
    
    cdef np.ndarray freqs = np.linspace(wi,wf,n)
    cdef int nX = datX.shape[0]
    cdef int i,j
    cdef float sumI = 0.0
    cdef float sumR = 0.0
    cdef np.ndarray[DTYPE_t,ndim=1] wD = np.zeros(n,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=2] out = np.zeros([2,n],dtype=DTYPE)
    cdef DTYPE_t value = 0.0
    
    for j in range(0,n):
        sumR = 0.0
        sumI = 0.0
        for i in range(0,nX):
            sumR = sumR + datY[i]*np.cos(2000*np.pi*datX[i]/freqs[j])
            sumI = sumI + datY[i]*np.sin(2000*np.pi*datX[i]/freqs[j])
        value = np.sqrt((sumR/nX)**2+(sumI/nX)**2)
        wD[j]=value
    out[0]=freqs
    out[1]=wD
    return out