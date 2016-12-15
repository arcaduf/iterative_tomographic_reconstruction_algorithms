import cython

import numpy as np
cimport numpy as np


cdef extern void gridrec_v4_backproj( float *S , int npix , int nang , float *angles , float *param ,
                                      float *lut , float *deapod , float *filt , float *I , char *fftwfn)

cdef extern void gridrec_v4_forwproj( float *S , int npix , int nang , float *angles , float *param , 
                                      float *lut , float *deapod , float *I , char *fftwfn)

cdef extern void create_fftw_wisdom_file(char *fn, int npix)


@cython.boundscheck( False )
@cython.wraparound( False )
def createFFTWWisdomFile( npix, \
                fftw_wisdom_file_name
                ):
    cdef char* cfn = fftw_wisdom_file_name
    create_fftw_wisdom_file(cfn,npix);

                    

@cython.boundscheck( False )
@cython.wraparound( False )
def backproj( np.ndarray[ float , ndim=2 , mode="c" ] sinogram not None ,
              np.ndarray[ float , ndim=1 , mode="c" ] angles not None ,
              np.ndarray[ float , ndim=1 , mode="c" ] param not None ,
              np.ndarray[ float , ndim=1 , mode="c" ] lut not None ,
              np.ndarray[ float , ndim=2 , mode="c" ] deapod not None ,
              np.ndarray[ float , ndim=1 , mode="c" ] filt = None ):

    cdef int nang, npix
    cdef char* cfn

    fftw_wisdom_file_name = None
    if fftw_wisdom_file_name==None:
        cfn = "/dev/null"
    else:
        cfn = "~/tomcat/Programs/pymodule_gridrec_v4/profile.wis"

    nang , npix = sinogram.shape[0], sinogram.shape[1]
    myFloat = sinogram.dtype

    image = np.zeros( ( npix , npix ) , dtype=myFloat, order='C' )
    
    cdef float [:,::1] cimage = image

    gridrec_v4_backproj( &sinogram[0,0] , npix , nang , &angles[0] , &param[0] ,
                      &lut[0] , &deapod[0,0] , &filt[0] , &cimage[0,0], cfn )

    return image



@cython.boundscheck( False )
@cython.wraparound( False )
def forwproj( np.ndarray[ float , ndim=2 , mode="c" ] image not None ,
              np.ndarray[ float , ndim=1 , mode="c" ] angles not None ,
              np.ndarray[ float , ndim=1 , mode="c" ] param not None ,  
              np.ndarray[ float , ndim=1 , mode="c" ] lut not None ,
              np.ndarray[ float , ndim=2 , mode="c" ] deapod not None ,    
              fftw_wisdom_file_name=None
               ):

    cdef int nang, npix
    cdef char* cfn
    if fftw_wisdom_file_name==None:
        cfn = "/dev/null"
    else:
        cfn = fftw_wisdom_file_name

    npix = image.shape[0]
    nang = len( angles )
    myFloat = image.dtype

    sino = np.zeros( ( nang , npix ) , dtype=myFloat, order='C' )
    
    cdef float [:,::1] csino = sino

    gridrec_v4_forwproj( &csino[0,0] , npix , nang , &angles[0] , &param[0] , 
                         &lut[0] , &deapod[0,0] , &image[0,0], cfn )

    return sino

