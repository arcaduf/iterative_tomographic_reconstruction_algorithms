import cython

import numpy as np
cimport numpy as np


cdef extern void nl_weights( float *I , float *param , float *cout0 ,int *cout1 , int *cout2 )

cdef extern void sbnltv( float *cu , float *I , float *W , int *Y , int *SY , float *param , float *cout0 )  




@cython.boundscheck( False )
@cython.wraparound( False )
def nl_weights_cy( np.ndarray[ float , ndim=2 , mode="c" ] image not None , 
                   np.ndarray[ float , ndim=1 , mode="c" ] param not None ):
    myfloat = image.dtype

    iNx = np.int( param[0] );  iNy = np.int( param[1] );  iNbNeigh = np.int( param[5] )
    iw = np.int( param[3] );  icn = np.int( param[6] )

    if iNbNeigh > iw * iw - 4:
        iNbNeigh = iw * iw
    else:
        if icn:
            iNbNeigh += 4
    iN3 = iNbNeigh * 2 * 2

    out0 = np.zeros( ( iNx * iNy * iN3 ) , dtype=myfloat, order='C' )
    cdef float [::1]cout0 = out0

    out1 = np.zeros( ( iNx * iNy * iN3 ) , dtype=np.int32, order='C' )
    cdef int [::1]cout1 = out1 

    out2 = np.zeros( ( iNx , iNy ) , dtype=np.int32, order='C' )
    cdef int [:,::1] cout2 = out2 

    nl_weights( &image[0,0] , &param[0] , &cout0[0] , &cout1[0] , &cout2[0,0] )

    return out0 , out1 , out2




@cython.boundscheck( False )
@cython.wraparound( False )
def sbnltv_cy( np.ndarray[ float , ndim=2 , mode="c" ] I not None , 
               np.ndarray[ float , ndim=1 , mode="c" ] W not None ,
               np.ndarray[ int , ndim=1 , mode="c" ] Y not None ,
               np.ndarray[ int , ndim=2 , mode="c" ] SY not None ,
               np.ndarray[ float , ndim=1 , mode="c" ] param not None ,  
             ):
    myfloat = I.dtype

    iNx = param[0];  iNy = param[1];  iNbNeigh = param[4];  iw = np.int( param[3] ) 

    if iNbNeigh > iw * iw - 4:
        iNbNeigh = iw * iw
    iN3 = iNbNeigh * 2 * 2      

    u = np.zeros( ( iNx , iNy ) , dtype=myfloat, order='C' )
    u[:] = I
    cdef float [:,::1] cu = u  

    out0 = np.zeros( ( iNx , iNy ) , dtype=myfloat, order='C' )
    cdef float [:,::1] cout0 = out0

    sbnltv( &cu[0,0] , &I[0,0] , &W[0] , &Y[0] , &SY[0,0] , &param[0] , &cout0[0,0] )

    return out0



