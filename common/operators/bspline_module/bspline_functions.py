import numpy as np
from scipy import misc
from scipy import signal
from scipy import ndimage  




####  CONSTANTS
eps = 1e-8




##########################################################
##########################################################
####                                                  ####
####        CHANGE FROM PIXEL TO BSPLINE BASIS        ####
####                                                  ####
##########################################################
########################################################## 

def pixel_basis_to_bspline( image , bspline_degree ):
    m = bspline_degree
    if m == 0:
        bspline_image = image
    elif m == 2:
        bspline_image = signal.qspline2d( image )
    elif m == 3:
        bspline_image = signal.cspline2d( image )
    else:
        sys.exit('\nERROR: B-spline degree not supported !!')
    return bspline_image




##########################################################
##########################################################
####                                                  ####
####  COMPUTE R^{n} OF B_SPLINE GENERAL SLOW FORMULA  ####
####                                                  ####
##########################################################
##########################################################

def positive_power( p ):
    def positive_power_p( x ):
        if np.float32( x ) > 0.0:
            return np.power( np.float32( x ) , p )
        else:
            return 0.0
    return positive_power_p 


def scalar_product(lamba, f):
    def spf(x):
        return float(lamba)*f(x)
    return spf


def finite_difference(f, h):
    def finite_difference_function(x):
        if h != 0.0:
            return ( f( x + 0.5*h ) - f( x - 0.5*h ) ) / np.float32( h )
        else:
            sys.exit('\nError in "finite_difference": h = 0 !!')
    return finite_difference_function 


def radon_n_bspline_general( y , theta , m , n ):
    exponent = 2*m - n + 1
    y_plus = positive_power( exponent )
    

    ##  Calculate  m+1 fold derivative (analytically)
    deriv_positive_power = scalar_product( ( misc.factorial( exponent ) / \
                   misc.factorial( exponent - (m +1) ) ) , positive_power( m - n ) )

    if theta == 0.0 or theta == np.pi/2.0 or theta == np.pi:
        y_plus = deriv_positive_power
        
    
    ##  Consider special case of theta = 0 , pi
    if np.abs( theta - np.pi/2.0 ) > eps:
        for i in range( m + 1 ):
            y_plus = finite_difference( y_plus, np.cos(theta) )
    

    ##  Consider special case of theta = pi/2
    if np.abs( theta ) > eps  and np.abs( theta - np.pi ) > eps:
        for i in range( m + 1 ):
            y_plus = finite_difference( y_plus , np.sin(theta) )
    
    return y_plus(y)/float( misc.factorial( exponent ) )  




##########################################################
##########################################################
####                                                  ####
####            INIT B-SPLINE LOOK UP TABLE           ####
####                                                  ####
##########################################################
########################################################## 

##  Formula n. 28 of M. Nilchian's paper:
##  "Fast iterative reconstruction of differential phase contrast
##   X-ray tomograms", M. Nilchian et al., Optics Express, 2013.
##
##  R^{n}{beta(x)}( y , theta ) = sum_{k1=0}^{m+1} sum_{k2=0}^{m+1} (-1)^{k1+k2} *
##      * comb( m+1 , k1 ) * comb( m+1 , k2 ) * ( y + ( (m+1)/2 - k1 )cos(theta) + 
##      ( (m+1)/2 - k2 )sin(theta) )_{+}^{2m-n+1} / [( 2m-n+1 )! * cos(theta)^{m+1} *
##      * sin(theta)^{m+1}]

def init_lut_bspline( nsamples_y , angles , bspline_degree , rt_degree , proj_support_y ):
    m = bspline_degree
    n = rt_degree    
    exponent = 2*m - n + 1

    
    ##  Define y-range as equally spaced points between -(m+1)/2, ... ,(m+1)/1 .
    ##  Correct is to adopt a rectangular support with length nsamples_y * sqrt(2) ,
    ##  but we do without (function values in edges are very small) .
    ##  nsamples_y should be even
    yrange = proj_support_y
    yarray = np.arange( nsamples_y ).astype( np.float32 )
    yarray -= nsamples_y / 2.0 
    yarray /= np.float32( nsamples_y )
    yarray *= yrange
    yarray_tile = yarray.reshape( 1 , yarray.shape[0] )
    
    
    ##  Define the theta-range as equally spaced points between 0 ... pi
    nsamples_theta = len( angles )
    theta_array_tile = angles.reshape( len( angles ) , 1 )
    
    
    ##  Repeat the tile yarray_tile vertically for nsamples_theta times
    y_matrix = np.tile( yarray_tile , ( nsamples_theta , 1 ) )


    ##  Repeat the tile theta_array_tile for nsamples_y times
    theta_matrix = np.tile( theta_array_tile , ( 1 , nsamples_y ) )


    ##  Precalculate sin and cos of all the angles
    sin_theta_matrix = np.sin( theta_matrix )
    cos_theta_matrix = np.cos( theta_matrix ) 

    
    ##  Prepare denominator matrix: fact( 2m-n+1 ) * cos(theta)^{m+1} * sin(theta)^{m+1}
    ##  Take care to reassign to angles 0 , pi/2 , pi values different from 0
    power_matrix = np.power( sin_theta_matrix * cos_theta_matrix , m+1 )
    divisor = np.float32( misc.factorial( exponent ) ) * power_matrix

    ind_0 = np.argwhere( np.abs( angles ) < eps )
    ind_90 = np.argwhere( np.abs( angles - np.pi/2.0 ) < eps )
    ind_180 = np.argwhere( np.abs( angles - np.pi ) < eps ) 

    if len( ind_0 ) !=0:
        divisor[ind_0, :] = 1.0

    if len( ind_90 ) != 0:
        divisor[ind_90, :] = 1.0
    
    if len( ind_180 ) != 0:
        divisor[ind_180, :] = 1.0

    
    ##  Allocate memory for the radon transform coefficients
    ##  of the B-splines functions
    result = np.zeros( ( nsamples_theta , nsamples_y ) )


    ##  Compute numerator matrix
    for k_1 in  range( 0,  m+1+1 ):
        for k_2 in  range(0, m+1+1 ):
            num = y_matrix + ( m/2.0 + 0.5 - k_1 ) * cos_theta_matrix + \
                  ( m/2.0 + 0.5 - k_2 ) * sin_theta_matrix
            num[ num < 0.0] = 0.0
            num_power = np.power( num , exponent )
            num_power *= np.power( -1.0 , k_1 + k_2 ) * misc.comb( m + 1 , k_1 ) * misc.comb( m + 1 , k_2 )
            result += num_power
    
    
    ##  Divide for the divisor which is indipendent from the sums on k_1 and k_2 
    result /= divisor


    ##  Correcting for the general values 0 , pi/2 , pi with slow general formula
    if len( ind_0 ) != 0 or len( ind_90 ) != 0 or len( ind_180 ) != 0:  
        for y in range( nsamples_y ):
            if len( ind_0 ) != 0:
                result[ ind_0 , y ] = radon_n_bspline_general( yarray[y] , 0.0 , m , n )
            if len( ind_90 ) != 0: 
                result[ ind_90 , y ] = radon_n_bspline_general( yarray[y] , np.pi/2.0 , m , n )
            if len( ind_180 ) != 0: 
                result[ ind_180 , y ] = radon_n_bspline_general( yarray[y] , np.pi , m , n )

    return result




##########################################################
##########################################################
####                                                  ####
####        CONVERT FROM B-SPLINE TO PIXEL BASIS      ####
####                                                  ####
##########################################################
##########################################################

def b_spline( x , degree ):
    h = 1.0
    f = positive_power( degree )
    for i in range(degree + 1):
        f = finite_difference( f , h )
    return f( x ) / float( misc.factorial( degree ) )


def tensor_bspline( x , y , m ):
    return b_spline(x, m)*b_spline(y, m)       


def calc_bspline_grid_points( m ):
    bspline_grid_points = np.zeros( ( m + 1 + 1 , m + 1 + 1 ) )
    for i in range(m + 1 + 1):
        y = i - (m + 1)/2.0
        for j in range(m + 1 + 1):
            x = j - (m + 1)/2.0
            bspline_grid_points[i,j] = tensor_bspline( x , y , m )
    return bspline_grid_points


def convert_from_bspline_to_pixel_basis( image_bspline , bspline_degree ):
    convolvent = calc_bspline_grid_points( bspline_degree )
    return ndimage.filters.convolve( image_bspline , convolvent ) 

