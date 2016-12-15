import cython

import numpy as np
import multiprocessing as mproc
cimport numpy as np


cdef extern void gen_forwproj( float* image , int npix , float* angles , int nang , float* lut ,
                   int lut_size , float support_bspline , int num_cores , float* sino )

cdef extern void gen_backproj( float* sino , int npix , float* angles , int nang , float* lut ,
                               int lut_size , float support_bspline , float* image )


                    

@cython.boundscheck( False )
@cython.wraparound( False )
def forwproj( np.ndarray[ float , ndim=2 , mode="c" ] image not None ,
              np.ndarray[ float , ndim=1 , mode="c" ] angles not None ,
              np.ndarray[ float , ndim=2 , mode="c" ] lut not None ,
              np.ndarray[ float , ndim=1 , mode="c" ] param not None ):

    cdef int nang, npix , lut_size , num_cores
    cdef float support_bspline

    npix = image.shape[0]
    nang = len( angles )
    myfloat = image.dtype
    lut_size = int( param[0] )
    support_bspline = np.float32( param[1] )

    num_cores = mproc.cpu_count()

    sino = np.zeros( ( nang , npix ) , dtype=myfloat, order='C' )
    
    cdef float [:,::1] csino = sino

    gen_forwproj( &image[0,0] , npix , &angles[0] , nang , &lut[0,0] ,
                  lut_size , support_bspline , num_cores , &csino[0,0] )
                                 
    return sino




@cython.boundscheck( False )
@cython.wraparound( False )
def backproj( np.ndarray[ float , ndim=2 , mode="c" ] sino not None ,
              np.ndarray[ float , ndim=1 , mode="c" ] angles not None ,
              np.ndarray[ float , ndim=2 , mode="c" ] lut not None ,
              np.ndarray[ float , ndim=1 , mode="c" ] param not None ):

    cdef int nang, npix

    nang , npix = sino.shape[0] , sino.shape[1]
    myfloat = sino.dtype
    lut_size = int( param[0] )
    support_bspline = np.float32( param[1] )

    image = np.zeros( ( npix , npix ) , dtype=myfloat, order='C' )
    
    cdef float [:,::1] cimage = image

    gen_backproj( &sino[0,0] , npix , &angles[0] , nang , &lut[0,0] ,
                  lut_size , support_bspline , &cimage[0,0] )
                                 
    return image
