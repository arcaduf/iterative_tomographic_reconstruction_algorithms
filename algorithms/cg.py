#####################################################################
#####################################################################
####                                                             #### 
####  CONJUGATE GRADIENT METHODS FOR TOMOGRAPHIC RECONSTRUCTION  ####
####                                                             ####
#####################################################################
#####################################################################
#
#   The problem is of the form:  Ax = b
#       A ---> forward projector
#       x ---> image to reconstruct
#       b ---> sinogram
#
#   Since for the CG, the matrix A should be a hermitian positive
#   definite, the problem is restated as:  Hx = c  ,  where:
#       H ---> A^{t}A
#       c ---> A^{t}b




####  PYTHON MODULES
from __future__ import division , print_function
import numpy as np
import pylab as py
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import bspline_functions as bfun




####  MY FORMAT VARIABLE
myfloat = np.float32




###########################################################
###########################################################
####                                                   #### 
####                     INNER PRODUCT                 ####
####                                                   ####
###########################################################
###########################################################  

def innpr( array1 , array2 ):
    return np.inner( np.array( array1 ).reshape(-1) , 
                     np.array( array2 ).reshape(-1) ) 




###########################################################
###########################################################
####                                                   #### 
####              COMPUTE FINITE DIFFERENCES           ####
####                                                   ####
###########################################################
###########################################################

def G( m ):
    return np.gradient( m )


def Gt( g ):
    nx , ny = g[0].shape
    Gy = np.gradient( np.diag( np.ones( nx ) ) )[1]
    Gx = np.gradient( np.diag( np.ones( ny ) ) )[0]
    m = np.dot( g[1] , Gx ) + np.dot( Gy , g[0] )
    return m



###########################################################
###########################################################
####                                                   #### 
####                  CONJUGATE GRADIENT               ####
####                                                   ####
###########################################################
###########################################################

def cg( x , b , tp , n_iter , ii , param , ind=None , plot=False ):
    ##  r_{0} = c - Hx_{0}
    r = tp.At( b ) - tp.At( tp.A( x ) )


    if ind is None:
        ind = 0
    
    i1 = ii[0];  i2 = ii[1]


    ##  p_{0} = r_{0}
    p = r.copy()
    Hp = np.zeros( ( x.shape[0] , x.shape[1] ) , dtype=myfloat )
    r_old = np.zeros( ( x.shape[0] , x.shape[1] ) , dtype=myfloat )


    ##  Set stopping criterion
    eps = 1e-12
    err = 1e12
    it = 0

    

    ##  Initialize plot
    if plot is True and ind == 0:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        Ln = ax.imshow( x[i2:i1:-1,i1:i2] , animated=True , cmap=cm.Greys_r )
        plt.ion()  

    

    ##  Start loop
    x_old = x.copy()
    info = []

    while it < n_iter: # and err > eps:
        ##  compute Hp_{k} = A^{t}Ap_{k}
        Hp[:,:] = tp.At( tp.A( p ) )


        ##  alpha_{k} = < r_{k} , r_{k} > / < p_{k} , Hp_{k} >
        alpha = innpr( r , r ) / ( innpr( p , Hp ) + np.spacing( 1 ) )


        ##  x_{k+1} = x_{k} + alpha_{k}p_{k}
        x[:,:] += alpha * p
        diff = x - x_old
        x_old = x.copy()


        ##  r_{k+1} = r_{k} - alpha_{k}Hp_{k}
        r_old[:,:] = r
        r -= alpha * Hp
        err = np.linalg.norm( r )

        if ind == 0:
            print('    CG iter. num.: ', it,'  --->  error: ', err)

        if err < eps and ind == 0:
            print('    Error < threshold at iter. num.: ', it)
            break


        ## p_{k+1} = r_{k+1} + <r_{k+1},r_{k+1}>/<r_{k},r_{k}> * p_{k}
        p[:,:] = r + innpr( r , r ) / ( innpr( r_old , r_old ) + np.spacing( 1 ) )* p


        ##  Update iteration index
        it +=1

        if param.checkit is True:
            if param.projector == 'grid-pswf' or param.projector == 'grid-kb':
                x_aux = x.copy()
            elif param.projector == 'radon':
                x_aux = x.copy() 
            elif param.projector == 'bspline':
                x_aux = bfun.convert_from_bspline_to_pixel_basis( x , 3 )
            if it < 10:
                niter = '00' + str( it )
            elif it < 100:
                niter = '0' + str( it )
            else:
                niter = str( it )
            io.writeImage( param.path_rmse + 'reco_iter' + niter + '.DMP' , x_aux )  

        err  = np.linalg.norm( r )
        obj  = np.linalg.norm( tp.A( x ) - b )**2
        diff = np.linalg.norm( x - x_old )
        info.append( [ err , diff , obj ] )

    
    return x , info




###########################################################
###########################################################
####                                                   #### 
####           CONJUGATE GRADIENT --- LASSO L1         ####
####                                                   ####
###########################################################
###########################################################

def cg_lasso_l1( x , b , tp , niter , v , lambd1 , mu , ind=None ):
    ##  r_{0} = c - Hx_{0}
    r = tp.At( b ) + v - ( tp.At( tp.A( x ) ) + ( lambd1 + mu ) * x )

    if ind is None:
        ind = 0


    ##  p_{0} = r_{0}
    p = r.copy()
    Hp = np.zeros( ( x.shape[0] , x.shape[1] ) , dtype=myfloat )
    r_old = np.zeros( ( x.shape[0] , x.shape[1] ) , dtype=myfloat )


    ##  Set stopping criterion
    eps = 1e-12
    err = 1e12;  it = 0


    ##  Start loop
    while it < niter and err > eps:
        ##  compute Hp_{k} = A^{t}Ap_{k}
        Hp[:,:] = tp.At( tp.A( p ) ) + ( lambd1 + mu ) * p


        ##  alpha_{k} = < r_{k} , r_{k} > / < p_{k} , Hp_{k} >
        alpha = innpr( r , r ) / ( innpr( p , Hp ) + np.spacing( 1 ) )


        ##  x_{k+1} = x_{k} + alpha_{k}p_{k}
        x[:,:] += alpha * p


        ##  r_{k+1} = r_{k} - alpha_{k}Hp_{k}
        r_old[:,:] = r
        r -= alpha * Hp
        err = np.linalg.norm( r )

        if ind == 0:
            print('    CG iter. num.: ', it,'  --->  error: ', err)

        if err < eps and ind == 0:
            print('    Error < threshold at iter. num.: ', it)
            break


        ## p_{k+1} = r_{k+1} + <r_{k+1},r_{k+1}>/<r_{k},r_{k}> * p_{k}
        p[:,:] = r + innpr( r , r ) / innpr( r_old , r_old ) * p


        ##  update iteration index
        it += 1


    return x




###########################################################
###########################################################
####                                                   #### 
####           CONJUGATE GRADIENT --- LASSO TV         ####
####                                                   ####
###########################################################
###########################################################

def cg_lasso_tv( x , b , tp , niter , v , lambd1 , mu , ind=None ):
    ##  r_{0} = c - Hx_{0}
    r = tp.At( b ) + Gt( v ) - ( tp.At( tp.A( x ) ) + lambd1 * x + Gt( G( x ) ) )

    if ind is None:
        ind = 0


    ##  p_{0} = r_{0}
    p = r.copy()
    Hp = np.zeros( ( x.shape[0] , x.shape[1] ) , dtype=myfloat )
    r_old = np.zeros( ( x.shape[0] , x.shape[1] ) , dtype=myfloat )


    ##  Set stopping criterion
    eps = 1e-12
    err = 1e12;  it = 0


    ##  Start loop
    while it < niter and err > eps:
        ##  compute Hp_{k} = A^{t}Ap_{k}
        Hp[:,:] = tp.At( tp.A( p ) ) + lambd1 * p + mu * Gt( G( p ) )


        ##  alpha_{k} = < r_{k} , r_{k} > / < p_{k} , Hp_{k} >
        alpha = innpr( r , r ) / ( innpr( p , Hp ) + np.spacing( 1 ) )


        ##  x_{k+1} = x_{k} + alpha_{k}p_{k}
        x[:,:] += alpha * p


        ##  r_{k+1} = r_{k} - alpha_{k}Hp_{k}
        r_old[:,:] = r
        r -= alpha * Hp
        err = np.linalg.norm( r )

        if ind == 0:
            print('    CG iter. num.: ', it,'  --->  error: ', err)

        #if err < eps and ind == 0:
        #    print('    Error < threshold at iter. num.: ', it)
        #    break


        ## p_{k+1} = r_{k+1} + <r_{k+1},r_{k+1}>/<r_{k},r_{k}> * p_{k}
        p[:,:] = r + innpr( r , r ) / innpr( r_old , r_old ) * p


        ##  update iteration index
        it += 1


    return x




###########################################################
###########################################################
####                                                   #### 
####        CONJUGATE GRADIENT --- PLUG AND PLAY       ####
####                                                   ####
###########################################################
###########################################################

def cg_plug_and_play( x , b , tp , niter , v , lambd1 , mu , ind=None ):
    ##  r_{0} = c - Hx_{0}
    r = tp.At( b ) + mu * v - ( tp.At( tp.A( x ) ) + ( lambd1 + mu ) * x )

    if ind is None:
        ind = 0


    ##  p_{0} = r_{0}
    p = r.copy()
    Hp = np.zeros( ( x.shape[0] , x.shape[1] ) , dtype=myfloat )
    r_old = np.zeros( ( x.shape[0] , x.shape[1] ) , dtype=myfloat )


    ##  Set stopping criterion
    eps = 1e-12
    err = 1e12;  it = 0


    ##  Start loop
    while it < niter and err > eps:
        ##  compute Hp_{k} = A^{t}Ap_{k}
        Hp[:,:] = tp.At( tp.A( p ) ) + ( lambd1 + mu ) * p


        ##  alpha_{k} = < r_{k} , r_{k} > / < p_{k} , Hp_{k} >
        alpha = innpr( r , r ) / ( innpr( p , Hp ) + np.spacing( 1 ) )


        ##  x_{k+1} = x_{k} + alpha_{k}p_{k}
        x[:,:] += alpha * p


        ##  r_{k+1} = r_{k} - alpha_{k}Hp_{k}
        r_old[:,:] = r
        r -= alpha * Hp
        err = np.linalg.norm( r )

        if ind == 0:
            print('    CG iter. num.: ', it,'  --->  error: ', err)

        if err < eps and ind == 0:
            print('    Error < threshold at iter. num.: ', it)
            break


        ## p_{k+1} = r_{k+1} + <r_{k+1},r_{k+1}>/<r_{k},r_{k}> * p_{k}
        p[:,:] = r + innpr( r , r ) / innpr( r_old , r_old ) * p


        ##  update iteration index
        it += 1


    return x




###########################################################
###########################################################
####                                                   #### 
####               TEST CONJUGATE GRADIENT             ####
####                                                   ####
###########################################################
###########################################################

def cg_test( A , x , b , n_iter=None , eps=None ):
    ##  r_{0} = c - Hx_{0}
    At = np.transpose( A )
    r = np.dot( At , b ) - np.dot( At , np.dot( A , x ) )


    ##  p_{0} = r_{0}
    p = r.copy()
    Hp = np.zeros( ( x.shape[0] , x.shape[1] ) , dtype=myfloat )
    r_old = np.zeros( ( x.shape[0] , x.shape[1] ) , dtype=myfloat )


    ##  Set stopping criterion
    if eps is None:
        eps = 1e-12
    else:
        n_iter = 1e5

    err = 1e12;  it = 0


    ##  Start loop
    while it < n_iter and err > eps:
        ##  compute Hp_{k} = A^{t}Ap_{k}
        Hp[:] = np.dot( At , np.dot( A , p ) )


        ##  alpha_{k} = < r_{k} , r_{k} > / < p_{k} , Hp_{k} >
        alpha = innpr( r , r ) / innpr( p , Hp )


        ##  x_{k+1} = x_{k} + alpha_{k}p_{k}
        x[:] += alpha * p


        ##  r_{k+1} = r_{k} - alpha_{k}Hp_{k}
        r_old[:] = r
        r -= alpha * Hp
        err = np.linalg.norm( r )

        print('    CG iter. num.: ', it,'  --->  error: ', err)

        if err < eps:
            print('    Error < threshold at iter. num.: ', it)
            break


        ## p_{k+1} = r_{k+1} + <r_{k+1},r_{k+1}>/<r_{k},r_{k}> * p_{k}
        p[:] = r + innpr( r , r ) / innpr( r_old , r_old ) * p


        ##  update iteration index
        it += 1


    return x




###########################################################
###########################################################
####                                                   #### 
####               TEST CONJUGATE GRADIENT             ####
####                                                   ####
###########################################################
###########################################################

def pcg_test( A , x , b , M , n_iter=None , eps=None ):
    ##  r_{0} = c - Hx_{0}
    At = np.transpose( A )
    prec = np.linalg.inv( M )
    r = np.dot( At , b ) - np.dot( At , np.dot( A , x ) )


    ##  z_{0} = M^{-1}r_{0}
    z = np.dot( prec , r )


    ##  p_{0} = z_{0}
    p = z.copy()
    Hp = np.zeros( ( x.shape[0] , x.shape[1] ) , dtype=myfloat )
    r_old = np.zeros( ( x.shape[0] , x.shape[1] ) , dtype=myfloat )
    z_old = np.zeros( ( x.shape[0] , x.shape[1] ) , dtype=myfloat ) 


    ##  Set stopping criterion
    if eps is None:
        eps = 1e-12
    else:
        n_iter = 1e5

    err = 1e12;  it = 0


    ##  Start loop
    while it < n_iter and err > eps:
        ##  compute Hp_{k} = A^{t}Ap_{k}
        Hp[:] = np.dot( At , np.dot( A , p ) )


        ##  alpha_{k} = < r_{k} , z_{k} > / < p_{k} , Hp_{k} >
        alpha = innpr( r , z ) / innpr( p , Hp )


        ##  x_{k+1} = x_{k} + alpha_{k}p_{k}
        x[:] += alpha * p


        ##  r_{k+1} = r_{k} - alpha_{k}Hp_{k}
        r_old[:] = r
        r -= alpha * Hp
        err = np.linalg.norm( r )

        print('    CG iter. num.: ', it,'  --->  error: ', err)

        if err < eps:
            print('    Error < threshold at iter. num.: ', it)
            break


        ##  z_{k+1} = M^{-1}r_{k+1}
        z_old[:] = z
        z[:] = np.dot( prec , r )


        ## p_{k+1} = r_{k+1} + <r_{k+1},r_{k+1}>/<r_{k},r_{k}> * p_{k}
        p[:] = r + innpr( r , z ) / innpr( r_old , z_old ) * p


        ##  update iteration index
        it += 1


    return x
