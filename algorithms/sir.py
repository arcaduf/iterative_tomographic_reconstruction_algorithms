#################################################################################
#################################################################################
#################################################################################
#######                                                                   #######
#######                STATISTICAL ITERATIVE RECONSTRUCTION:              #######
#######                                                                   #######
#######        Author: Filippo Arcadu, arcusfil@gmail.com, 12/02/2015     #######
#######                                                                   #######
#################################################################################
#################################################################################
#################################################################################




####  PYTHON MODULES
from __future__ import division,print_function
import time
import datetime
import argparse
import os
import sys
import glob
import numpy as np
from scipy import ndimage as nd
import multiprocessing as mproc




####  PYTHON PLOTTING MODULES
import pylab as py
import matplotlib.pyplot as plt
import matplotlib.cm as cm   




####  MY PYTHON MODULES
cpath = '../common/'
sys.path.append( cpath + 'myimage/' )
sys.path.append( cpath + 'operators/regularization_module/' )
import my_image_io as io
import my_image_process as proc
import my_image_display as dis
import penalty as reg
import utils




####  CLASS TOMOGRAPHIC PROJECTORS
import class_projectors_grid as cpj1
import class_projectors_bspline as cpj2
import class_projectors_radon as cpj3




####  CLASS SIR PARAMETERS
import class_sir_param as csp




####  MY FORMAT VARIABLES
myfloat = np.float32
myint = np.int




####  LIST PROJECTIONS
list_pr = [ 'grid-kb' , 'grid-pswf' , 'bspline' , 'radon' , 
            'pix-driv' , 'ray-driv' , 'dist-driv' , 'slant' ]




##########################################################
##########################################################
####                                                  ####
####             GET INPUT ARGUMENTS                  ####
####                                                  ####
##########################################################
##########################################################

def getArgs():
    parser = argparse.ArgumentParser(description='''Statistical iterative reconstruction
                                                    based on the regridding operators with minimal
                                                    oversampling''',
                          formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-Di', '--pathin', dest='pathin', default='./',
                        help='Specify path to input data')    
    
    parser.add_argument('-i', '--filein', dest='filein',
                        help='Specify name of input sinogram')
    
    parser.add_argument('-Do', '--pathout', dest='pathout',
                        help='Specify path to output data') 
    
    parser.add_argument('-o', '--fileout', dest='fileout',
                        help='Specify name of output reconstruction')
    
    parser.add_argument('-a', '--algorithm', dest='algorithm', default = 'em',
                        help='Specify the statistical algorithm you want to use:'
                        + 'em ---> expectation-maximization'
                        + 'sps ---> separable paraboloidal surrogate')    
    
    parser.add_argument('-g', '--geometry', dest='geometry', default='0',
                        help='Specify projection geometry;'
                             +' -g 0 --> equiangular projections between 0 and 180 degrees (default);'
                             +' -g angles.txt use a list of angles (in degrees) saved in a text file')     
    
    parser.add_argument('-c', '--center', dest='ctr', default=-1, type=myfloat,
                        help='Centre of rotation (default: center of the image);')

    parser.add_argument('-z', '--edge_padding', type=myfloat, default=0.0,
                        help='Enable edge padding of the sinogram')  
    
    parser.add_argument('-p', dest='plot', action='store_true',
                        help='Display check-plots during the run of the code')
    
    parser.add_argument('-n', dest='n_iter', type=myint,
                        help='Specify number of iterations')

    parser.add_argument('-beta', dest='reg_cost', type=myfloat, default=0.1,
                        help='Specify regularization constant')  

    parser.add_argument('-r', dest='regularization',
                        help='Specify type of regularization:'
                             + ' "h" or "huber"    ---> Huber penalty;'
                             + ' "t" or "tikhonov" ---> l2 penalty'
                             + ' "a" or "haar"     ---> l1 penalty')

    parser.add_argument('-pr', dest='projector', default='grid-pswf',
                        help='Select projectors: grid-pswf -- grid-kb ' + \
                             '-- bspline (cubic) -- radon (lin)' + \
                             '-- pix-driv (pixel-driven linear)' + \
                             '-- ray-driv (ray-driven)' + \
                             '-- dist-driv (distance-driven)' + \
                             '-- slant (slant-stacking)')

    parser.add_argument('-hc', dest='huber_cost', type=myfloat, default=0.1,
                        help='Specify delta for Huber regularization') 

    parser.add_argument('-l', dest='logfile', action='store_true',
                        help='Write log file')

    parser.add_argument('-init', dest='init_object', action='store_true',
                        help='Initialize reconstruction with FBP reconstruction') 

    parser.add_argument('-cit', dest='checkit',
                        help='Enable saving every iteration for futher analysis')

    parser.add_argument('-eps', dest='eps', type=myfloat,
                        help='Select stopping epsilon') 

    parser.add_argument('-lt', dest='lt', action='store_true',
                        help='Enable local tomography configuration')

    parser.add_argument('-nc', dest='num_cores', type=np.int, default=-1,
                        help='Choose how many cores to use for parallel \
                              computations in 3D. Examples: -nc -1 --> use \
                              all available cores; -nc 2 --> use 2 cores')  
    
    args = parser.parse_args()
    
    if args.n_iter is None and args.eps is None:
        parser.print_help()
        sys.exit('\nERROR: Stopping criterion not specified!\n')

    if args.projector not in list_pr:
        parser.print_help()
        sys.exit('\nERROR: Selected projector ' + args.pr + ' not available!\n' + \
                 'Select among: ' + str( list_pr ) + '\n' ) 

    return args




###########################################################
###########################################################
####                                                   #### 
####    MAXIMUM-LIKELIHOOD EXPECTATION-MINIMIZATION    ####
####                                                   ####
###########################################################
###########################################################

def forward( tp , x ):
    return tp.A( x )




def backward( tp , b ):
    return tp.At( b )




def em( x , b , tp , param ):
    ##  Get number pixels, views, ctr, stopping criterion
    nz , m , n = b.shape
    eps = param.eps
    n_iter = param.n_iter
    b = b.astype( myfloat )


    
    ##  Pre-compute constant part of the denominator
    ##  ( A^{T} 1 )^{T}
    ones = np.ones( ( m , n ) , dtype=myfloat )
    d_cost = tp.At( ones )
    d_cost[d_cost==0] = d_cost[d_cost==0] + 1e-10


    
    ##  Initialize useful arrays
    b_new = b.copy()
    err_list = []


    
    ##  Reconstruction indices
    if param.edge_padding != 0.0:
        i1 = param.index_start
        i2 = param.index_end
    else:
        i1 = 0
        i2 = n  



    ##  Initialize plot
    if nz == 1:
        nzz = 0
    else:
        nzz = np.int( nz * 0.5 )  

    if param.plot is True:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        Ln = ax.imshow( x[nzz,i2:i1:-1,i1:i2] , animated=True , cmap=cm.Greys_r )
        plt.ion()   



    ##  Start reconstruction loop
    it = 0
    err = 1e20
    x_aux = x.copy()

    while it < n_iter and err > eps:
        ##  Compute forward projection of new iterate
        x_old = x.copy()

        if param.num_cores == -1:
            pool = mproc.Pool()
        else:
            pool = mproc.Pool( param.num_cores )
        results = [ pool.apply_async( forward , args=( tp , x[i,:,:] ) ) \
                    for i in range( nz ) ]
        b_new[:] = np.array( [ res.get() for res in results ] )
        pool.close()
        pool.join()        
        
        b_new[b_new==0] = b_new[b_new==0] + np.spacing(1)
        b_aux = b / b_new

        if param.num_cores == -1:
            pool = mproc.Pool()
        else:
            pool = mproc.Pool( param.num_cores )         
        results = [ pool.apply_async( backward , args=( tp , b_aux[i,:,:] ) ) \
                    for i in range( nz ) ]
        x_aux[:] = np.array( [ res.get() for res in results ] )
        pool.close()
        pool.join() 

        x[:] *= ( 1.0 / d_cost ) * x_aux


        ##  Compute step improvement
        err = np.sum( ( b_new[nzz,:,:] - b[nzz,:,:] )**2 ) / np.sum( b[nzz,:,:]**2 )
        err1 = np.linalg.norm( x[nzz,:,:] - x_old[nzz,:,:] )
        err2 = 0.5 * np.linalg.norm( tp.A( x[nzz,:,:] ) - b[nzz,:,:] )        
        err_list.append( [ err1 , err2 ] )
        it += 1

        print('\nEM-iteration: ', it,'   error: ', err)

        
        ##  Plot intermediate reconstruction as check
        if param.plot is True:
            ax.imshow( x[nzz,i2:i1:-1,i1:i2] , animated=True , cmap=cm.Greys_r )  
            py.draw()
            plt.pause(1)


        ##  Write iterate for further analysis of the algorithm
        if param.checkit is True:
            if param.projector == 'grid-pswf' or param.projector == 'grid-kb' or param.projector == 'pix-driv':
                x_aux = x[nzz,i1:i2,i1:i2].copy()
            elif param.projector == 'radon':
                x_aux = x[nzz,i2:i1:-1,i2:i1:-1].copy() 
            elif param.projector == 'bspline':
                x_aux = bfun.convert_from_bspline_to_pixel_basis( x[nzz,i2:i1:-1,i2:i1:-1] , 3 )
            if it < 10:
                niter = '00' + str( it )
            elif it < 100:
                niter = '0' + str( it )
            else:
                niter = str( it )  
            io.writeImage( param.path_rmse + 'reco_iter' + niter + '.DMP' , x_aux )  

    info = np.array( err_list )

    if param.plot is True:
        py.ioff()
        
    return x , info




###########################################################
###########################################################
####                                                   #### 
####          SEPARABLE PARABOLOIDAL SURROGATE         ####
####                                                   ####
###########################################################
########################################################### 



def sobel( x , gamma ):
    gx = nd.filters.sobel( x , axis=0 , mode='reflect' )
    gy = nd.filters.sobel( x , axis=1 , mode='reflect' )
    g  = np.sqrt( gx**2 + gy**2 )
    return g




def sps( x , b , tp , param ):
    ##  Get number pixels, views, ctr, stopping criterion
    nz , m , n = b.shape
    eps = param.eps
    n_iter = param.n_iter
    b = b.astype( myfloat )


    
    ##  Regularization parameters
    beta = param.reg_cost

    if param.reg == 'huber':
        delta = param.huber_cost

    
    
    ##  Pre-compute weights
    b_max = np.max( b );  b_den = np.sum( b**2 ) 
    w = np.exp( - b / b_max ).astype( myfloat )


    
    ##  Pre-compute deninator:
    ##  d = ( y * gamma )^{T} * A
    aux = tp.A( np.ones( ( n , n ) , dtype=myfloat ) )
    aux = w * np.kron( np.ones( ( nz , 1 , 1 ) , dtype=myfloat ) , aux )

    if param.num_cores == -1:
        pool = mproc.Pool()
    else:
        pool = mproc.Pool( param.num_cores )     
    results = [ pool.apply_async( backward , args=( tp , aux[i,:,:] ) ) \
                for i in range( nz ) ]
    den = np.array( [ res.get() for res in results ] )
    pool.close()
    pool.join()

    
    
    ##  Initialize useful arrays
    b_diff = b.copy();  num = den.copy();  
    R_num = np.zeros( ( nz , n , n ) , dtype=myfloat )
    R_den = np.zeros( ( nz , n , n ) , dtype=myfloat )
    g     = np.zeros( ( nz , n , n ) , dtype=myfloat ) 
    err_list = []


    
    ##  Reconstruction indices
    if param.edge_padding != 0.0:
        i1 = param.index_start
        i2 = param.index_end
    else:
        i1 = 0
        i2 = n  



    ##  Initialize plot
    if nz == 1:
        nzz = 0
    else:
        nzz = np.int( nz * 0.5 )

    if param.plot is True:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        Ln = ax.imshow( x[nzz,i2:i1:-1,i1:i2] , animated=True , cmap=cm.Greys_r )         
        plt.ion()        
                    


    ##  Start reconstruction loop
    it = 0
    err = 1e20
    b_aux = b.copy()

    while it < n_iter and err > eps:
        ##  Compute forward projection of new iterate
        if param.num_cores == -1:
            pool = mproc.Pool()
        else:
            pool = mproc.Pool( param.num_cores )         
        results = [ pool.apply_async( forward , args=( tp , x[i,:,:] ) ) \
                    for i in range( nz ) ]
        b_aux[:] = np.array( [ res.get() for res in results ] )
        pool.close()
        pool.join() 

        b_diff[:] = b_aux - b

        
        ##  Compute numerator
        b_diff *= w

        if param.num_cores == -1:
            pool = mproc.Pool()
        else:
            pool = mproc.Pool( param.num_cores )         
        results = [ pool.apply_async( backward , args=( tp , b_diff[i,:,:] ) ) \
                    for i in range( nz ) ]
        num[:] = np.array( [ res.get() for res in results ] )
        pool.close()
        pool.join() 

        
        ##  Regularization
        gamma = 0
        if param.reg == 'huber':
            if param.num_cores == -1:
                pool = mproc.Pool()
            else:
                pool = mproc.Pool( param.num_cores ) 
            results = [ pool.apply_async( sobel , args=( x[i,:,:] , gamma ) ) \
                        for i in range( nz ) ] 
            g[:] = np.array( [ res.get() for res in results ] )
            pool.close()
            pool.join()
            
            if param.num_cores == -1:
                pool = mproc.Pool()
            else:
                pool = mproc.Pool( param.num_cores ) 
            results = [ pool.apply_async( reg.huber_num , args=( g[i,:,:] , delta ) ) \
                        for i in range( nz ) ] 
            R_num[:] = np.array( [ res.get() for res in results ] )
            pool.close()
            pool.join()
            
            if param.num_cores == -1:
                pool = mproc.Pool()
            else:
                pool = mproc.Pool( param.num_cores ) 
            results = [ pool.apply_async( reg.huber_den , args=( g[i,:,:] , delta ) ) \
                        for i in range( nz ) ] 
            R_den[:] = np.array( [ res.get() for res in results ] )
            pool.close()
            pool.join()             

        
        elif param.reg == 'tikhonov':
            if param.num_cores == -1:
                pool = mproc.Pool()
            else:
                pool = mproc.Pool( param.num_cores ) 
            results = [ pool.apply_async( reg.tikhonov_num , args=( x[i,:,:] , gamma ) ) \
                        for i in range( nz ) ] 
            R_num[:] = np.array( [ res.get() for res in results ] )
            pool.close()
            pool.join()
            
            if param.num_cores == -1:
                pool = mproc.Pool()
            else:
                pool = mproc.Pool( param.num_cores ) 
            results = [ pool.apply_async( reg.tikhonov_den , args=( x[i,:,:] , gamma ) ) \
                        for i in range( nz ) ] 
            R_den[:] = np.array( [ res.get() for res in results ] )
            pool.close()
            pool.join() 

        
        elif param.reg == 'haar':
            if param.num_cores == -1:
                pool = mproc.Pool()
            else:
                pool = mproc.Pool( param.num_cores ) 
            results = [ pool.apply_async( reg.haar_num , args=( x[i,:,:] , gamma ) ) \
                        for i in range( nz ) ] 
            R_num[:] = np.array( [ res.get() for res in results ] )
            pool.close()
            pool.join()
            
            if param.num_cores == -1:
                pool = mproc.Pool()
            else:
                pool = mproc.Pool( param.num_cores ) 
            results = [ pool.apply_async( reg.haar_den , args=( x[i,:,:] , gamma ) ) \
                        for i in range( nz ) ] 
            R_den[:] = np.array( [ res.get() for res in results ] )
            pool.close()
            pool.join()  


        ##  Update reconstruction
        x_old = x.copy()
        x[:] -= ( num + beta * R_num ) / ( den + beta * R_den + np.spacing(1) )

        
        ##  Enforce positivity
        if param.edge_padding is False:
            x[:] = np.clip( x , 0 , np.max( x ) )
        
        ##  Compute step improvement
        if param.reg == 'huber':
            gc = g[0,:,:].copy()
            R = gc.copy()
            R[ gc <= delta ] = 0.5 * gc[ gc <= delta ]**2
            R[ gc > delta ]  = delta * gc[ gc > delta ] - 0.5 * delta**2
            R = np.sum( R )
        elif param.reg == 'tikhonov':
            R = np.linalg.norm( x[0,:,:] )**2
        else:
            R = 0

        err = np.sum( ( b_diff[nzz,:,:] )**2 ) / b_den
        err1 = np.linalg.norm( x[nzz,:,:] - x_old[nzz,:,:] )
        err2 = 0.5 * np.linalg.norm( tp.A( x[nzz,:,:] ) - b[nzz,:,:] ) + beta * R
        err_list.append( [ err1 , err2 ] )
        it += 1

        print('\nSPS-iteration: ', it,'   error: ', err)

        
        ##  Plot intermediate reconstruction as check
        if param.plot is True:
            ax.imshow( x[nzz,i2:i1:-1,i1:i2] , animated=True, cmap=cm.Greys_r )  
            plt.draw()
            plt.pause(1)


        ##  Write iterate for further analysis of the algorithm
        if param.checkit is True:
            if param.projector == 'grid-pswf' or param.projector == 'grid-kb':
                x_aux = x[0,i1:i2,i1:i2].copy()
            elif param.projector == 'radon':
                x_aux = x[0,i2:i1:-1,i2:i1:-1].copy() 
            elif param.projector == 'bspline':
                x_aux = bfun.convert_from_bspline_to_pixel_basis( x[0,i2:i1:-1,i2:i1:-1] , 3 )
            if it < 10:
                niter = '00' + str( it )
            elif it < 100:
                niter = '0' + str( it )
            else:
                niter = str( it )  
            io.writeImage( param.path_rmse + 'reco_iter' + niter + '.DMP' , x_aux ) 



    ##  Conversion for bspline reconstruction and rotate
    if param.projector == 'bspline':
        for i in range( nz ):         
            x[i,:,:] = bfun.convert_from_bspline_to_pixel_basis( x[i,:,:] , 3 )
        x[:,:,:] = x[:,::-1,::-1]       


    info = np.array( err_list )

    if param.plot is True:
        py.ioff()
        
    return x , info   




###########################################################
###########################################################
####                                                   #### 
####                        SIR                        ####
####                                                   ####
###########################################################
###########################################################

def sir( b , a , param ):
    ##  Get number of pixels and angles
    m , n , nz = param.nang , param.npix_op , param.nz
    b  = np.array( b ).astype( myfloat )


    
    ##  Init forward and back-projector
    if param.projector == 'grid-pswf':
        tp = cpj1.projectors( n , a , kernel='pswf' , oversampl=2.0 )
    elif param.projector == 'grid-kb':
        tp = cpj1.projectors( n , a , kernel='kb' , oversampl=1.5 , 
                              W=6.6, errs=6.0e-6, interp='lin' )
    elif param.projector == 'bspline':
        tp = cpj2.projectors( n , a , param ,  bspline_degree=3 , 
                              proj_support_y=4 )
    elif param.projector == 'radon':
        tp = cpj2.projectors( n , a , param , bspline_degree=1 , 
                              proj_support_y=2 )
    elif param.projector == 'pix-driv':
        tp = cpj3.projectors( n , a , oper='pd' ) 
    elif param.projector == 'ray-driv':
        tp = cpj3.projectors( n , a , oper='rd' ) 
    elif param.projector == 'dist-driv':
        tp = cpj3.projectors( n , a , oper='dd' )
    elif param.projector == 'slant':
        tp = cpj3.projectors( n , a , oper='ss' ) 


    
    ##  Initialize x
    x = np.ones( ( nz , n , n ) , dtype=myfloat )

    if param.init_object is True:
        i1 = param.index_start
        i2 = param.index_end
        x_new = [];  b_new = []

        for i in range( nz ):
            x[i,:,:] = tp.fbp( b[i,:,:] )
            if param.plot is True:
                dis.plot( x[i,:,:] , 'Initialization' )



    ##  Reconstruction with CG
    if param.algorithm == 'em':
        print( '\n\nUsing Maximum-Likelihood Expectation-Maximization' )
        x[:] , info = em( x , b , tp , param ) 


    ##  Reconstruction with Lasso-L1
    elif param.algorithm == 'sps':
        print( '\n\nUsing Separable Paraboloidal Surrogate' ) 
        x[:] , info = sps( x , b , tp , param )


    
    
    ##  Conversion for bspline reconstruction and rotate
    if param.projector == 'bspline':
        for i in range( nz ):
            x[i,:,:] = bfun.convert_from_bspline_to_pixel_basis( x[i,:,:] , 3 )
        x[:,:,:] = x[:,::-1,::-1]

    
    return x , info  




##########################################################
##########################################################
####                                                  ####
####                SAVE RECONSTRUCTION               ####
####                                                  ####
##########################################################
##########################################################

def save_reco( pathout , filein , args , param , reco ):
    if args.fileout is not None:
        filename = pathout + args.fileout

    else:        
        filename = filein
        extension = filename[len(filename)-4:] 
        filename = filename[:len(filename)-4]
        filename += param.root
        filename += extension
        filename = pathout + filename

    io.writeImage( filename , reco )

    print('\nReconstruction saved in:\n', filename)
                                                        



##########################################################
##########################################################
####                                                  ####
####                    WRITE LOG FILE                ####
####                                                  ####
##########################################################
##########################################################

def write_logfile( pathin , pathout , args , angles , ctr , param , time_tot , info ):
    filename = args.sino
    filename = filename[:len(filename)-4] + '_logfile.txt' 

    fp = open( pathout + filename , 'w' )
    fp.write('STATISTICAL ITERATIVE RECONSTRUCTION')
    today = datetime.date.today()
    fp.write('\n\nReconstruction performed on ' + str( today ) )
    fp.write('\n\nInput file:\n' + pathin + args.sino )
    fp.write('\n\nProjection angles:\n' + str( angles ))
    fp.write('\n\nCenter of rotation axis: ' + str( ctr ))
    if args.edge_padding is True:
        fp.write('\n\nEdge padding enabled')
    fp.write('\n\nNumber of iterations:\n' + str( param.n_iter ))
    fp.write('\n\nStopping threshold:\n' + str( param.eps ))
    fp.write('\n\nRegularization:\n' + str( param.reg ))   
    fp.write('\n\nReconstruction algorithm: ' + args.algorithm)
    fp.write('\n\nTime elapsed for the reconstruction: ' + str( time_tot ) )
    n_iter = len( info )
    fp.write('\n\nNumber of iterations: ' + str( n_iter ))
    fp.write('\n\nIteration error list:')
    for i in range( n_iter ):
        fp.write('\n' + str( info[i] )) 
    fp.close()




##########################################################
##########################################################
####                                                  ####
####                    WRITE LOG FILE                ####
####                                                  ####
##########################################################
##########################################################

def write_info( pathout , filein , info , param , args ):
    n = len( info )
    info = np.array( info )
    info1  = np.array( info[:,1] ).reshape( n , 1 )
    info2  = np.array( info[:,0] ).reshape( n , 1 )

    filename = filein
    extension = filename[len(filename)-4:] 
    filename = filename[:len(filename)-4] + param.root

    if args.reco is None:
        file1 = pathout + filename + '_object_score.txt'
        file2 = pathout + filename + '_diff_score.txt'
    else:
        fileout = args.reco
        fileout = fileout[:len(fileout)-4]
        file1 = pathout + fileout + '_object_score.txt'
        file2 = pathout + fileout + '_diff_score.txt' 

    np.savetxt( file1 , info1 , fmt='%.5e', delimiter='\n' )
    np.savetxt( file2 , info2 , fmt='%.5e', delimiter='\n' )




##########################################################
##########################################################
####                                                  ####
####                       MAIN                       ####
####                                                  ####
##########################################################
##########################################################

def main():
    ##  Initial print
    print('\n')
    print('##########################################################')   
    print('##########################################################')
    print('#####                                                #####')
    print('#####       STATISTICAL ITERATIVE RECONSTRUCTION     #####')
    print('#####                                                #####')     
    print('##########################################################')
    print('##########################################################')       
    print('\n')




    ##  Getting arguments
    args = getArgs()

    
    
    ##  Get input & output paths
    pathin , pathout = utils.get_io_path( args )  
    
    print('\nInput path: \n', pathin)
    print('\nOutput path:\n', pathout)



    ##  Get input sinogram
    sino_list = [] ; filein = []
    
    if args.filein is not None:
        sinoname = pathin + args.filein  
        sino = io.readImage( sinoname ).astype( myfloat )
        nang , npix = sino.shape
        nz = 1
        sino_list.append( sino )
        filein.append( args.filein )
        print('\nSinogram to reconstruct:\n', sinoname)
    
    else:
        print('\nReading stack of images\n' )
        curr_dir = os.getcwd()
        os.chdir( pathin )
        
        for f in os.listdir('./'):
            if f.endswith( '.DMP' ) is True:
                ext = '.DMP'
                break
            elif f.endswith( '.tif' ) is True:
                ext = '.tif'
                break
            else:
                sys.exit('\nERROR: no .DMP or .tif file found in:\n' + pathin)

        filein.append( sorted( glob.glob( '*' + ext ) ) )
        nz = len( filein[0] )
        os.chdir( curr_dir )

        print('Stack extension: ', ext)
        print('Number of slices: ', nz)
        
        print('\nLoading images .... ')
        for i in range( nz ):
            if i == 0:
                sino = io.readImage( pathin + filein[0][i] ).astype( myfloat )
                nang , npix = sino.shape
                sino_list.append( sino )
            else:
                sino_list.append( io.readImage( pathin + filein[0][i] ).astype( myfloat ) )
        print( ' done! ' )               

    print('\nNumber of projection angles: ', nang)
    print('Number of pixels: ', npix )



    ##  Check plot
    if args.plot is True:
        if nz == 1:
            dis.plot( sino_list[0] , 'Input sinogram' )
        else:
            nzz = np.int( 0.5 * nz )
            dis.plot( sino_list[nzz] , 'Input sinogram' )   


    
    ##  Center of rotation axis
    if args.ctr == -1:
        ctr = npix * 0.5
    
    elif args.ctr != -1:
        ctr = args.ctr



    
    ##  Enable edge padding
    if args.lt is True:
        edf = 0.87
    elif args.lt is False and args.edge_padding != 0:
        edf = args.edge_padding
    else:
        edf = 0.0

    if edf:
        npix_old = npix
        for i in range( nz ):
            sino_list[i] = proc.sino_edge_padding( sino_list[i] , edf )
        npix = sino_list[0].shape[1]
        i1 = myint( ( npix - npix_old ) * 0.5 )
        i2 = i1 + npix_old
        ctr += i1

        print('\nEdge padding: ', edf )
        print('Number of edge-padded pixels: ', npix)
        print('Index start: ', i1,'   Index end: ', i2)
        print('Center of rotation axis new position: ', ctr)

        if args.plot is True:
            dis.plot( sino_list[0] , 'Sinogram with edge-padding' )

    else:
        npix_old = npix
        i1 = 0
        i2 = npix    



    ##  Correct for the center of rotation axis    
    if args.ctr != -1:
        for i in range( nz ):
            sino_list[i] = proc.sino_correct_rot_axis( sino_list[i] , ctr )
    
    print('\nCenter of rotation axis position: ', ctr)
    print('Center of rotation corrected')
    
    


    ##  Get geometry
    if args.geometry == '0':
        angles = utils.create_projection_angles( nang )
    else:
        angles = utils.create_projection_angles( textfile=pathin + args.geometry )



    ##  Setup iterative procedure
    print('\nSetup of the iterative procedure:')

    if nz == 0:
        labelout = pathout + filein[0][ : len( filein[0] ) - 4 ]
    else:
        labelout = pathout + filein[0][0][ : len( filein[0] ) - 4 ]  
    
    param = csp.sir_param( nang , npix_old , nz , ctr , labelout , args )

    print( 'Selected projector: ' , args.projector )

    if args.eps is not None:
        print('Stopping threshold: ', param.eps )

    if args.n_iter is not None:
        print('Number of iterations: ', param.n_iter)

    if args.plot is True:
        print('Interactive plot:', param.plot)

    if args.logfile is True:
        print('Interactive plot:', param.logfile)  

    if param.reg is not None:
        if param.reg == 'huber':
            print('Regularization type: Huber penalty')
            print('Huber constant ---> delta: ')
        elif param.reg == 'tikhonov':
            print('Regularization type: l2 penalty' )
        elif param.reg == 'haar':
            print('Regularization type: l1 penalty' )

        print('Regularization constant (beta): ', param.reg_cost)

    if args.init_object is True:
        print('Initialization with FBP reconstruction:', param.init_object)  



    ##  Reconstruction
    print('\n\nPerforming STASTICAL ITERATIVE RECONSTRUCTION ....')
    time1 = time.time()
    reco_list , info = sir( sino_list , angles , param )
    time2 = time.time()
    print('.... reconstruction done!') 



    for i in range( nz ):
        ##  Crop reconstruction if edge-padding enabled
        if edf != 0.0:
            reco = reco_list[i,i1:i2,i1:i2]
        else:
            reco = reco_list[i,:,:] 


        ##  Show reconstruction
        if args.plot is True and nz==1:
            dis.plot( reco , 'Reconstruction' )
        elif args.plot is True and i==nzz:
            dis.plot( reco , 'Reconstruction' )

    
        ##  Save reconstruction
        if nz == 1:
            save_reco( pathout , filein[0] , args , param , reco )
        else:
            save_reco( pathout , filein[0][i] , args , param , reco )



    ##  Time elapsed for the reconstruction
    time_tot = ( time2 - time1 ) / 60.0
    print('\nTime elapsed for the reconstruction: ', time_tot)
    
    
    
    ##  Write log file
    if args.logfile is True:
        write_logfile( pathin , pathout , args , angles ,
                       ctr , param , time_tot , info )

        write_info( pathout , filein[0] , info , param , args ) 


    
    ##  Final print
    print('\n')
    print('\n')
    print('##########################################################')   
    print('##########################################################')
    print('#####                                                #####')
    print('#####   STATISTICAL ITERATIVE RECONSTRUCTION DONE!   #####')
    print('#####                                                #####')     
    print('##########################################################')
    print('##########################################################')       
    print('\n')    




##########################################################
##########################################################
####                                                  ####
####                    CALL TO MAIN                  ####
####                                                  ####
##########################################################
##########################################################   

if __name__ == '__main__':
    main()
