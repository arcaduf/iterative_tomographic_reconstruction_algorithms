##########################################################
##########################################################
####                                                  ####
####       SINOGRAM FILTERING FOR BACKPROJECTION      ####
####                                                  ####
##########################################################
##########################################################




####  PYTHON MODULES
import sys
import numpy as np




####  MY FORMAT VARIABLES
myfloat   = np.float32
myint     = np.int
mycomplex = np.complex64




##########################################################
##########################################################
####                                                  ####
####              CALCULATE FILTERING ARRAY           ####
####                                                  ####
##########################################################
##########################################################

def calc_filter( nfreq , ftype='ramp' , dpc=False ):
    ##  Any filtering for backprojection
    if ftype != 'none':
        ##  Half ramp ftypeer
        if dpc is False:
            filtarr = 2 * np.arange( nfreq + 1 ) / myfloat( 2 * nfreq )
            w = 2 * np.pi * np.arange( nfreq + 1 ) / myfloat( 2 * nfreq )
            d = 1.0
        
        ##  Half Hilbert ftypeer
        else:
            filtarr = np.ones( nfreq + 1 , dtype=mycomplex ) * ( 1.0j ) / ( 2 * np.pi ) 

            
        ##  Superimposing  noise-dumping ftypeers    
        if ftype == 'shepp-logan':
            filtarr[1:] *= np.sin( w[1:] ) / ( 2.0 * d * 2.0 * d * w[1:] )

        elif ftype == 'cosine':
            filtarr[1:] *= np.cos( w[1:] ) / ( 2.0 * d * w[1:] )  

        elif ftype == 'hamming':
            filtarr[1:] *= ( 0.54 + 0.46 * np.cos( w[1:] )/d )

        elif ftype == 'hanning':
            filtarr[1:] *= ( 1.0 + np.cos( w[1:]/d )/2.0 )


        ##  Compute second half of ftypeer for absorp. reconstr.
        if dpc is False:
            filtarr = np.concatenate( ( filtarr , filtarr[nfreq-1:0:-1] ) , axis=0 )
        
        ##  Compute second half of ftypeer for dpc reconstr.
        else:
            filtarr = np.concatenate( ( filtarr , np.conjugate( filtarr[nfreq-1:0:-1] ) ) , axis=0 ) 


    ##  No ftypeering
    else:
        filtarr = np.ones( 2 * nfreq )

    return filtarr      




##########################################################
##########################################################
####                                                  ####
####                 PROJECTION FILTERING             ####
####                                                  ####
##########################################################
##########################################################

def filter_proj( sino , ftype='ramp' , dpc=False ):
    ##  Compute oversamples array length
    nang , npix = sino.shape
    nfreq = 2 * int( 2**( int( np.ceil( np.log2( npix ) ) ) ) )


    ##  Compute filtering array
    filtarr = calc_filter( nfreq , ftype=ftype , dpc=dpc )


    ##  Zero-pad projections
    sino_filt = np.concatenate( ( sino , np.zeros( ( nang , 2*nfreq - npix ) ) ) , axis=1 ) 


    ##  Filtering in Fourier space
    for i in range( nang ):
        sys.stdout.write( 'Filtering projection number %d\r' % ( i + 1 , ) )
        sys.stdout.flush()
        sino_filt[i,:] = np.real( np.fft.ifft( np.fft.fft( sino_filt[i,:] ) * filtarr ) )


    ##  Replace values in the original array
    sino[:,:] = sino_filt[:,:npix]

    return sino

