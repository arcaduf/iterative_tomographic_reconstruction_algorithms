###############################################################################
###############################################################################
###############################################################################
######                                                                   ######
######                       DENOISING TOOLS                             ######
######                                                                   ######
######                         1) rof                                    ######
######                         2) gaussian                               ######
######                         3) median                                 ######
######                         4) bilateral                              ######
######                         5) tv_breg                                ######
######                         6) tv_chamb                               ######
######                         7) nl_means                               ######
######                         9) hosvd_den                              ######
######                        10) tgv_fast                               ######
######                        11) nltv_sb                                ######
######                                                                   ######
######        Author: Filippo Arcadu, arcusfil@gmail.com, 11/07/2013     ######
######                                                                   ######
###############################################################################
###############################################################################
###############################################################################




####  PYTHON MODULES
from __future__ import division, print_function
import sys
import numpy as np
import scipy
import skimage
from scipy import ndimage as ndi
from skimage import img_as_float
from skimage import restoration as rst

try:
    import sktensor as ten
    from sktensor import tucker as tuc
    tensor_module_avail = True
except ImportError:
    tensor_module_avail = False

from pymodule_split_bregman_nltv import split_bregman_nltv as sbnltv




####  MY VARIABLE TYPE
myfloat  = np.float64
myfloat2 = np.float32




###########################################################
###########################################################
####                                                   ####
####                   Preprocessing                   ####
####                                                   ####
###########################################################
###########################################################

def change_range( image ):
    img_min = np.min( image )
    img_max = np.max( image )
    image[:,:] =  2.0 / ( img_max - img_min ) * ( image - img_min ) - 1
    return img_as_float( image ) , img_min , img_max - img_min 



def restore_range( image , img_min , delta ):
    image[:,:] =  delta / 2.0 * ( image + 1 ) + img_min
    return image 




###########################################################
###########################################################
####                                                   ####
####         Rudin-Fatemi-Osher Total Vatiation        ####
####                                                   ####
###########################################################
###########################################################

def rof( im , weight=100 , eps=2.e-4 , n_iter_max=200 ):
    px = np.zeros_like(im)
    py = np.zeros_like(im)
    pz = np.zeros_like(im)
    gx = np.zeros_like(im)
    gy = np.zeros_like(im)
    gz = np.zeros_like(im)
    d = np.zeros_like(im)
    i = 0
    while i < n_iter_max:
        d = - px - py - pz
        d[1:] += px[:-1] 
        d[:, 1:] += py[:, :-1] 
        d[:, :, 1:] += pz[:, :, :-1] 
        
        out = im + d
        E = (d**2).sum()

        gx[:-1] = np.diff(out, axis=0) 
        gy[:, :-1] = np.diff(out, axis=1) 
        gz[:, :, :-1] = np.diff(out, axis=2) 
        norm = np.sqrt(gx**2 + gy**2 + gz**2)
        E += weight * norm.sum()
        norm *= 0.5 / weight
        norm += 1.
        px -= 1./6.*gx
        px /= norm
        py -= 1./6.*gy
        py /= norm
        pz -= 1/6.*gz
        pz /= norm
        E /= float(im.size)
        if i == 0:
            E_init = E
            E_previous = E
        else:
            if np.abs(E_previous - E) < eps * E_init:
                break
            else:
                E_previous = E
        i += 1
    return out



 
###########################################################
###########################################################
####                                                   ####
####                 Gaussian filtering                ####
####                                                   ####
###########################################################
###########################################################

def gaussian( noisy , sigma_value ):
    return ndi.gaussian_filter( noisy , 2.0 )




###########################################################
###########################################################
####                                                   ####
####                 Gaussian filtering                ####
####                                                   ####
###########################################################
###########################################################

def median( noisy , radius=2 ):

    return ndi.median_filter( noisy , radius )




###########################################################
###########################################################
####                                                   ####
####                 Bilateral denoising               ####
####                                                   ####
###########################################################
###########################################################

def bilateral( noisy , win_size=5 , sigma_range=None , sigma_spatial=1 ,
               bins=10000 , mode='constant' , cval=0 ):

    return rst.denoise_bilateral( noisy , win_size=win_size,
                                 sigma_range=sigma_range,
                                 sigma_spatial=sigma_spatial,
                                 bins=bins, mode=mode, cval=cval )




###########################################################
###########################################################
####                                                   ####
####                Bregman Total Variation            ####
####                                                   ####
###########################################################
###########################################################  

def tv_breg( noisy , weight=50 , max_iter=100 , eps=0.001 , isotropic=True ):
    noisy[:] , m , delta = change_range( noisy )
    denois = rst.denoise_tv_bregman( noisy , weight , max_iter=max_iter ,
                                     eps=eps , isotropic=isotropic )
    denois[:] = restore_range( denois , m , delta )
    return denois





###########################################################
###########################################################
####                                                   ####
####              Chambolle Total Variation            ####
####                                                   ####
###########################################################
########################################################### 

def tv_chamb( noisy , weight=50 , eps=0.0002 , n_iter_max=200 , multichannel=False ):
    noisy[:] , m , delta = change_range( noisy )   
    denois = rst.denoise_tv_chambolle( noisy , weight=weight , eps=eps ,
                                       n_iter_max=n_iter_max ,
                                       multichannel=multichannel )
    denois[:] = restore_range( denois , m , delta )
    return denois  




###########################################################
###########################################################
####                                                   ####
####                    Non local means                ####
####                                                   ####
###########################################################
###########################################################

def nl_means( noisy , patch_size=7 , patch_distance=11 , h=0.1 , multichannel=False ,
              fast_mode=True ):

    noisy[:] , m , delta = change_range( noisy ) 
    denois = rst.nl_means_denoising( noisy , patch_size=patch_size ,
                                     patch_distance=patch_distance ,
                                     h=h , multichannel=multichannel ,
                                     fast_mode=fast_mode )
    denois[:] = restore_range( denois , m , delta )
    return denois 




###########################################################
###########################################################
####                                                   ####
####  Denoising through HOSVD: dictionary repr. in 3D  ####
####                                                   ####
###########################################################
###########################################################  

def core2curve( core ):
    curve = core.reshape( -1 )
    signs = np.sign( curve )
    curve = np.abs( curve )
    order = np.argsort( curve )
    curve = np.array( sorted( curve ) )
    return curve , order , signs



def curve2core( curve , s , order , signs ):
    core = np.zeros( s[0] * s[1] * s[2] , dtype=myfloat )
    core[order] = curve
    core *= signs
    core = core.reshape( s[0] , s[1] , s[2] )
    return core



def hosvd( vol_in , vol_ref , strength=1.0 ):
    ##  Check if package is installed
    if tensor_module_avail is False:
        sys.exit('\nERROR: sktensor module not available!\n')



    ##  Tucker decomposition of reference and noisy volumes
    n1 , n2 , n3 = vol_ref.shape
    R_C , R_U = tuc.hooi( ten.dtensor( vol_ref ) , [ n1 , n2 , n3 ] , init='nvecs' )

    n1 , n2 , n3 = vol_in.shape
    I_C , I_U = tuc.hooi( ten.dtensor( vol_in ) , [ n1 , n2 , n3 ] , init='nvecs' )
    mmax = np.max( vol_in )



    ##  Get curve of the ordered absolute values of the core
    ##  tensor from both reference and input volumes
    curve_ref , order , signs = core2curve( R_C )
    curve_in  , order , signs = core2curve( I_C )


                                                               
    ##  Adjust curve of the input volume to that of the
    ##  reference
    curve_den = curve_in + ( curve_ref - curve_in ) * strength
    core_den  = curve2core( curve_den , [ n1 , n2 , n3 ]  , order , signs )



    ##  Reconstruction
    vol_in[:] = ten.ttm( ten.dtensor( core_den ) , I_U )
    vol_in[:] = vol_in / np.max( vol_in ) * mmax

    return vol_in




###########################################################
###########################################################
####                                                   ####
####      Fast total generalized total variation       ####
####                                                   ####
###########################################################
###########################################################

def zero_pad(image, shape, position='corner'):
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape, dtype=int)

    if np.alltrue(imshape == shape):
        return image

    if np.any(shape <= 0):
        raise ValueError("ZERO_PAD: null or negative shape given")

    dshape = shape - imshape
    if np.any(dshape < 0):
        raise ValueError("ZERO_PAD: target size smaller than source one")

    pad_img = np.zeros(shape, dtype=image.dtype)

    idx, idy = np.indices(imshape)

    if position == 'center':
        if np.any(dshape % 2 != 0):
            raise ValueError("ZERO_PAD: source and target shapes "
                             "have different parity.")
        offx, offy = dshape // 2
    else:
        offx, offy = (0, 0)

    pad_img[idx + offx, idy + offy] = image

    return pad_img



def psf2otf(psf, shape):
    ##  Adapted from MATLAB psf2otf function
    if np.all(psf == 0):
        return np.zeros_like(psf)

    inshape = psf.shape
    psf = zero_pad(psf, shape, position='corner')


    ##  Circularly shift OTF so that the 'center' of the PSF is
    ##  [0,0] element of the array
    for axis, axis_size in enumerate(inshape):
        psf = np.roll(psf, -int(axis_size / 2), axis=axis)


    ##  Compute the OTF
    otf = np.fft.fft2(psf)

    ##  Estimate the rough number of operations involved in the FFT
    ##  and discard the PSF imaginary part if within roundoff error
    ##  roundoff error  = machine epsilon = sys.float_info.epsilon
    ##  or np.finfo().eps
    n_ops = np.sum(psf.size * np.log2(psf.shape))
    otf = np.real_if_close(otf, tol=n_ops)

    return otf

   

def diff_circ( I , dim , ori ):
    ##  Array shape
    nx , ny = I.shape

    ##  Vertical case
    if dim == 0:
        if ori == 'forward':
            J = np.diff( np.vstack( ( I , I[0,:] ) ) , axis=0 )
        elif ori == 'backward':
            J = -np.diff( np.vstack( ( I[nx-1,:] , I ) ) , axis=0 )
        else:
            J = np.diff( np.vstack( ( I , I[0,:] ) ) , axis=0 )

    ##  Horizontal case		
    else:
        if ori == 'forward':
            J = np.diff( np.hstack( ( I , I[:,0].reshape( nx , 1 ) ) ) , axis=1 )
        elif ori == 'backward': 
            J = -np.diff( np.hstack( ( I[:,ny-1].reshape( nx , 1 ) , I ) ) , axis=1 )
        else:
            J = np.diff( np.hstack( ( I , I[:,0].reshape( nx , 1 )  ) ) , axis=1 )

    return J             



def opt_X( Z1h , Z1v , U1h , U1v , Z2h , Z2d , Z2v , U2h , U2d , U2v , fkx ,
           fky , fkxc , fkyc , fI , A , rho , eta ):
    FZU1h = rho * np.fft.fft2( Z1h - U1h )
    FZU1v = rho * np.fft.fft2( Z1v - U1v )
    FZU2h = eta * np.fft.fft2( Z2h - U2h )
    FZU2d = eta * np.fft.fft2( Z2d - U2d )
    FZU2v = eta * np.fft.fft2( Z2v - U2v )

    B1 = fI + fkxc * FZU1h + fkyc * FZU1v
    B2 = -FZU1h + fkx * FZU2h + fky * FZU2d
    B3 = -FZU1v + fkx * FZU2d + fky * FZU2v

    FX  = A[0] * B1 + A[1] * B2 + A[2] * B3
    FQu = A[3] * B1 + A[4] * B2 + A[5] * B3
    FQv = A[6] * B1 + A[7] * B2 + A[8] * B3

    X  = np.real( np.fft.ifft2( FX ) )
    Qh = np.real( np.fft.ifft2( FQu ) )
    Qv = np.real( np.fft.ifft2( FQv ) )

    return X , Qh , Qv



def shrinkage( t , arr ):
    n = len( arr )

    T = arr[0]
    N = T * T 

    for i in range( 1 , n ):
        T = arr[i]
        N += T * T

    N = np.sqrt( N )
    S = np.clip( 1 - t / N , 0 , np.max( 1 - t / N ) )

    for i in range( n ):
        arr[i] = S * arr[i]

    return arr



def tgv_fast( I , alpha=0.06 , beta=0.05 , niter=20 ):
    ##  Image shape and normalization
    I = I.astype( myfloat2 ) 
    sx , sy = I.shape
    I[:] , m , delta = change_range( I ) 


    ##  ADMM parameters 
    rho = 1.0
    eta = 1.0


    ##  Differential kernels
    kx = np.array( [ 1 , -1 , 0 ] ).reshape( 1 , 3 )
    ky = kx.copy().reshape( 3 , 1 )

    fkx = psf2otf( kx , [sx,sy] )
    fky = psf2otf( ky , [sx,sy] )

    fkxc = np.conjugate( fkx )
    fkyc = np.conjugate( fky )

    fk = fkxc * fkx + fkyc * fky


    ##  Fourier transform of the input image
    fI = np.fft.fft2( I )


    ##  Pixel-wise inverse by using the image-wise conjugate method
    A11 = 1 + rho * fk;  A12 = -rho * fkxc;  A13 = -rho * fkyc
    A21 = -rho * fkx;  A22 = rho + eta * fk;  A23 = eta * fky * fkxc
    A31 = -rho * fky;  A32 = eta * fkx * fkyc;  A33 = rho + eta * fk    

    detA = 1.0 / ( (A11*A22*A33) + (A12*A32*A13) + (A31*A12*A23) - (A11*A32*A23) - (A31*A22*A13) - (A21*A12*A33) )

    iA11 = ( A22 * A33 - A23 * A32 ) * detA
    iA12 = ( A13 * A32 - A12 * A33 ) * detA
    iA13 = ( A12 * A23 - A13 * A22 ) * detA
    iA21 = ( A23 * A31 - A21 * A33 ) * detA
    iA22 = ( A11 * A33 - A13 * A31 ) * detA
    iA23 = ( A13 * A21 - A11 * A23 ) * detA
    iA31 = ( A21 * A32 - A22 * A31 ) * detA
    iA32 = ( A12 * A31 - A11 * A32 ) * detA
    iA33 = ( A11 * A22 - A12 * A21 ) * detA

    A = [ iA11 , iA12 , iA13 , iA21 , iA22 , iA23 , iA31 , iA32 , iA33 ]

    del fk, A11, A12, A13, A21, A22, A23, A31, A32, A33


    ##  Initialization
    O = I.copy()

    Z1h = diff_circ( I , 1 , 'forward' )
    Z1v = diff_circ( I , 0 , 'forward' )

    Z2h = np.zeros( ( sx , sy ) , dtype=myfloat )
    Z2d = np.zeros( ( sx , sy ) , dtype=myfloat )
    Z2v = np.zeros( ( sx , sy ) , dtype=myfloat )

    U1h = np.zeros( ( sx , sy ) , dtype=myfloat )
    U1v = np.zeros( ( sx , sy ) , dtype=myfloat )
    U2h = np.zeros( ( sx , sy ) , dtype=myfloat )
    U2d = np.zeros( ( sx , sy ) , dtype=myfloat )
    U2v = np.zeros( ( sx , sy ) , dtype=myfloat )


    ##  ADMM
    for t in range( niter ):
        #print( '\n\tTGV-fast iteraztion n.' , t )
        ##  Solve J
        X, Qh, Qv = opt_X( Z1h , Z1v , U1h , U1v , Z2h , Z2d , Z2v , U2h , U2d , U2v , fkx ,
                           fky , fkxc , fkyc , fI , A , rho , eta )

        ##  Solve Z1
        Xu = diff_circ( X , 1 , 'forward' );
        Xv = diff_circ( X , 0 , 'forward' );
	
        T1h = Xu - Qh
        T1v = Xv - Qv
	
        Z1h = T1h + U1h
        Z1v = T1v + U1v

        arr = shrinkage( alpha/rho , [ Z1h, Z1v ] )
        Z1h = arr[0];  Z1v = arr[1]

        ##  Solve Z2
        T2h = diff_circ( Qh, 2, 'backward' )
        T2d = diff_circ( Qh, 1, 'backward' ) + diff_circ( Qv, 2, 'backward' )
        T2v = diff_circ( Qv, 1, 'backward' )
	
        Z2h = T2h + U2h
        Z2d = T2d + U2d
        Z2v = T2v + U2v

        arr = shrinkage( beta/eta, [ Z2h, Z2d, Z2v ] )
        Z2h = arr[0];  Z2d = arr[1];  Z2v = arr[2]; 

        ##  Update U1
        U1h = U1h + ( T1h - Z1h )
        U1v = U1v + ( T1v - Z1v )
	
        ##  Update U2
        U2h = U2h + ( T2h - Z2h )
        U2d = U2d + ( T2d - Z2d )
        U2v = U2v + ( T2v - Z2v ) 


    ##  Restore image range
    X[:] = restore_range( X , m , delta ) 

    return X




###########################################################
###########################################################
####                                                   ####
####      Split Bregman non local total variation      ####
####                                                   ####
###########################################################
###########################################################

##  ps     --->  patch size
##  ws     --->  window search size
##  h      --->  parameter for weight function
##  nn     --->  number of neighbours
##  icn    --->  include close neighbours
##  mu     --->  ADMM parameter 1
##  lambd  --->  ADMM parameter 2
##  niter1 --->  ADMM number of outer iterations
##  niter2 --->  ADMM number of inner iterations

def nltv_sb( I , ps=5 , ws=11 , h=0.25**2 , nn=10 , icn=1 , mu=70 , lambd=20.0 ,
             niter1=4 , niter2=2 ):
    ##  Get image shape and normalize in [0,1]             
    nx , ny = I.shape
    I = I.astype( myfloat2 )
    I[:] , m , delta = change_range( I )


    ##  Create non local TV weights
    param_weights = np.array( [ nx , ny , ps , ws , h , nn , icn ] ).astype( myfloat2 )
    W , Y , SY = sbnltv.nl_weights_cy( I , param_weights )
    SYt = SY.copy();  SYt[:] = np.transpose( SY )
    SYt = SYt.astype( np.int32 );  Y = Y.astype( np.int32 )  


    ##  Run ADMM for split-Bregman NLTV
    param_admm = np.array( [ nx , ny , ps , ws , nn , lambd , mu , niter1 ,
                             niter2 , icn ] ).astype( myfloat2 )
    Inew = sbnltv.sbnltv_cy( I , W , Y , SYt , param_admm );


    ##  Restore original image range
    Inew[:] = restore_range( Inew , m , delta ) 

    return Inew
