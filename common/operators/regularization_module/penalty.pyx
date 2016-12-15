import cython

import numpy as np
cimport numpy as np


cdef extern void huber( float *I , int nr , int nc , float delta , char* term , float *R )
cdef extern void tikhonov( float *I , int nr , int nc , float *R )
cdef extern void haar( float *I , int nr , int nc , float *R ) 




@cython.boundscheck( False )
@cython.wraparound( False )
def huber_num( np.ndarray[ float , ndim=2 , mode="c" ] image not None , 
               float delta ):
    nr , nc = image.shape[0] , image.shape[1]
    myfloat = image.dtype

    penalty = np.zeros( ( nr , nc ) , dtype=myfloat, order='C' )
    cdef float [:,::1] cpenalty = penalty

    huber( &image[0,0] , nr , nc , delta , 'nom' , &cpenalty[0,0] )

    return penalty




@cython.boundscheck( False )
@cython.wraparound( False )
def huber_den( np.ndarray[ float , ndim=2 , mode="c" ] image not None , 
               float delta ):
    nr , nc = image.shape[0] , image.shape[1]
    myfloat = image.dtype


    penalty = np.zeros( ( nr , nc ) , dtype=myfloat, order='C' )
    cdef float [:,::1] cpenalty = penalty 

    huber( &image[0,0], nr , nc , delta , 'den' , &cpenalty[0,0] )

    return penalty



@cython.boundscheck( False )
@cython.wraparound( False )
def tikhonov_num( np.ndarray[ float , ndim=2 , mode="c" ] image not None , 
                  float gamma ):
    nr , nc = image.shape[0] , image.shape[1]
    myfloat = image.dtype

    penalty = np.zeros( ( nr , nc ) , dtype=myfloat, order='C' )
    cdef float [:,::1] cpenalty = penalty

    tikhonov( &image[0,0] , nr , nc , &cpenalty[0,0] )

    return penalty




@cython.boundscheck( False )
@cython.wraparound( False )
def tikhonov_den( np.ndarray[ float , ndim=2 , mode="c" ] image not None , 
                  float gamma ):
    nr , nc = image.shape[0] , image.shape[1]
    myfloat = image.dtype

    penalty = np.zeros( ( nr , nc ) , dtype=myfloat, order='C' )
    penalty[:] = 2

    return penalty




@cython.boundscheck( False )
@cython.wraparound( False )
def haar_num( np.ndarray[ float , ndim=2 , mode="c" ] image not None , 
              float gamma ):
    nr , nc = image.shape[0] , image.shape[1]
    myfloat = image.dtype

    penalty = np.zeros( ( nr , nc ) , dtype=myfloat, order='C' )
    cdef float [:,::1] cpenalty = penalty

    haar( &image[0,0] , nr , nc , &cpenalty[0,0] )

    return penalty




@cython.boundscheck( False )
@cython.wraparound( False )
def haar_den( np.ndarray[ float , ndim=2 , mode="c" ] image not None , 
              float gamma ):
    nr , nc = image.shape[0] , image.shape[1]
    myfloat = image.dtype

    penalty = np.zeros( ( nr , nc ) , dtype=myfloat, order='C' )

    return penalty








