import cython

import numpy as np
cimport numpy as np


cdef extern void radon_pd( float* sino , int npix , float* angles , int nang ,
                           int oper , int method , float* image )


                    

@cython.boundscheck( False )
@cython.wraparound( False )
def forwproj( np.ndarray[ float , ndim=2 , mode="c" ] image not None ,
              np.ndarray[ float , ndim=1 , mode="c" ] angles not None ,
              np.int method ):

    cdef int nang , npix , oper

    npix = image.shape[0]
    nang = len( angles )
    myfloat = image.dtype

    oper = 0

    if np.max( angles ) > 2 * np.pi:
        angles *= 2 * np.pi / 180.0

    angles = np.fft.fftshift( angles )

    if method is None:
        method = 0

    sino = np.zeros( ( nang , npix ) , dtype=myfloat, order='C' )
    
    cdef float [:,::1] csino = sino

    radon_pd( &image[0,0] , npix , &angles[0] , nang , oper , method , &csino[0,0] )
                                 
    return sino




@cython.boundscheck( False )
@cython.wraparound( False )
def backproj( np.ndarray[ float , ndim=2 , mode="c" ] sino not None ,
              np.ndarray[ float , ndim=1 , mode="c" ] angles not None ,
              np.int method ):

    cdef int nang , npix , oper

    nang , npix = sino.shape[0] , sino.shape[1]
    myfloat = sino.dtype

    oper = 1

    if np.max( angles ) > 2 * np.pi:
        angles *= 2 * np.pi / 180.0

    angles = np.fft.fftshift( angles ) 

    if method is None:
        method = 0  

    image = np.zeros( ( npix , npix ) , dtype=myfloat, order='C' )
    
    cdef float [:,::1] cimage = image

    radon_pd( &cimage[0,0] , npix , &angles[0] , nang , oper , method , &sino[0,0] )
                                 
    return image
