########################################################################
########################################################################
####                                                                #### 
####               Algebraic reconstruction algorithms              ####
####                                                                ####
####     Author: Filippo Arcadu, arcusfil@gmail.com, 21/07/2016     ####  
####                                                                ####
########################################################################
########################################################################




####  PYTHON MODULES
from __future__ import division , print_function
import numpy as np
import sys
import os
import shutil
import glob
import time
import argparse
from scipy import ndimage
import multiprocessing as mproc
import datetime




####  PYTHON MOFULES FOR PLOTTING
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as anim
import pylab as py




####  MY PYTHON MODULES
cpath = '../common/'
sys.path.append( cpath + 'myimage/' )
sys.path.append( cpath + 'operators/bspline_module/' )    
sys.path.append( cpath + 'operators/regularization_module/' )
import my_image_io as io
import my_image_process as proc
import my_image_display as dis
import my_image_denoise as den
import utils




####  CLASS TOMOGRAPHIC PROJECTORS
import class_projectors_grid as cpj1
import class_projectors_bspline as cpj2
import class_projectors_radon as cpj3




####  CLASS SIR PARAMETERS
import class_art_param as cap




####  MY FORMAT VARIABLES
myfloat = np.float32
myint = np.int




####  CONSTANTS
scale = 1e3
epsc   = 1e-10




###########################################################
###########################################################
####                                                   #### 
####                 GET INPUT ARGUMENTS               ####
####                                                   ####
###########################################################
###########################################################      

def getArgs():
    parser = argparse.ArgumentParser(description='Algebraic recontruction algorithms',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-Di', '--pathin', dest='pathin', default='./',
                        help='Specify path to input data')    
    
    parser.add_argument('-i', '--filein', dest='filein',
                        help='Specify name of input sinogram')
    
    parser.add_argument('-Do', '--pathout', dest='pathout',
                        help='Specify path to output data')

    parser.add_argument('-o', '--reco', dest='reco',
                        help='Specify name of output reconstruction')  
    
    parser.add_argument('-g', '--geometry', dest='geometry',default='0',
                        help='Specify projection geometry;'
                             +' -g 0 --> equiangular projections between 0 and 180 degrees;'
                             +' -g angles.txt use a list of angles (in degrees) saved in a text file')

    parser.add_argument('-c', dest='ctr', default=-1, type=myfloat,
                        help='Centre of rotation (default: center of the image);')

    parser.add_argument('-z', dest='edge_padding', type=myfloat, default=0.0,
                        help='Enable edge padding of the sinogram')

    parser.add_argument('-lt', dest='lt', action='store_true',
                        help='Enable local tomography configuration') 

    parser.add_argument('-n', dest='n_iter', type=myint,
                        help='Specify number of iterations')

    parser.add_argument('-a', dest='algorithm', default='sirt',
                        help='Specify algorithm: sirt') 

    parser.add_argument('-pr', dest='projector', default='grid-kb',
                        help='Select projectors: grid-pswf -- grid-kb ' + \
                             '-- bspline (cubic) ' + \
                             '-- slant (lin) ' + \
                             '-- pix-driv (pixel-driven linear) ' + \
                             '-- ray-driv (ray-driven) ' + \
                             '-- dist-driv (distance-driven)' )  

    parser.add_argument('-p', dest='plot', action='store_true',
                        help='Plot reconstruction')

    parser.add_argument('-l', dest='logfile', action='store_true',
                        help='Write log file') 

    parser.add_argument('-init', dest='init_object', action='store_true',
                        help='Initialize reconstruction with FBP reconstruction')

    parser.add_argument('-s', dest='scale', action='store_true',
                        help='Enable scaling') 

    parser.add_argument('-d', dest='dictionary',
                        help='Specify input file of the global dictionary') 

    parser.add_argument('-pc', dest='pc', type=myfloat, default=-1.0,
                        help='Select a down threshold to zero-out pixel values;'
                            + ' -pc -1 ---> phys. constr. disabled, '
                            + ' -pc alpha in [0.0,1.0] ---> phy. constr. on neg. values,'
                            + ' -pc alpha > 1.0 ---> phys. constr. on both neg.'
                            + ' values and outer-resol-circle points')

    parser.add_argument('-tv', dest='tv', type=myfloat, default=0.0,
                        help='Total variation' ) 

    parser.add_argument('-mask', dest='mask',
                        help='Input object support' )

    parser.add_argument('-mask_add', dest='mask_add',
                        help='Input additional masks for a priori knowledge' ) 

    parser.add_argument('-dpc', dest='dpc', action='store_true',
                        help='Enable DPC reconstruction')

    parser.add_argument('-dbp', dest='dbp', action='store_true',
                        help='Enable DBP reconstruction')

    parser.add_argument('-sg', dest='sg', type=myint, default=4,
                        help='Define Savitzky-Golay filter length')

    parser.add_argument('-cit', dest='checkit',
                        help='Enable saving every iteration for futher analysis')

    parser.add_argument('-eps', dest='eps', type=myfloat,
                        help='Select stopping epsilon')

    parser.add_argument('-z1', dest='inner_padding', type=myfloat, default=0.0,
                        help='Enable edge padding of the sinogram')

    parser.add_argument('-pi', dest='angle_pi', action='store_true',
                        help='Eliminate last projection')

    parser.add_argument('-nc', dest='num_cores', type=np.int, default=-1,
                        help='Choose how many cores to use for parallel \
                              computations in 3D. Examples: -nc -1 --> use \
                              all available cores; -nc 2 --> use 2 cores')   

    args = parser.parse_args()

    if args.n_iter is None and args.eps is None:
        parser.print_help()
        sys.exit('\nERROR: Stopping criterion not specified!\n')

    return args




###########################################################
###########################################################
####                                                   #### 
####  SIMULTANEOUS ITERATIVE RECONSTRUCTION TECHNIQUE  ####
####                       SIRT                        ####
####                                                   ####
###########################################################
###########################################################

def sirt_step( x , b , tp , R , C ):
    x[:] = x + C * tp.At( R * ( b - tp.A( x ) ) ) 
    return x



def sirt( x , b , tp , param ):       
    ##  Parameters
    n      = param.npix_op 
    nz     = param.nz
    it     = 0
    err    = 1e20
    info   = []


    ##  Create matrix R:  r_{ii} = 1 / ( sum_{j} a_{ij} )
    one_img = np.ones( ( n , n ) , dtype=myfloat )
    R       = tp.A( one_img )
    R[R==0] = R[R==0] + epsc
    R       = 1.0 / R


    ##  Create matrix C:  c_{jj} = 1 / ( sum_{i} a_{ij}  )
    one_sin = np.ones( ( param.nang , n ) , dtype=myfloat )
    C       =  tp.At( one_sin )
    C[C==0] = C[C==0] + epsc
    C       = 1.0 / C


    ##  Set stopping criterion                 
    if param.eps is None:
        eps = 1e-10
    else:
        eps = param.eps

    if param.n_iter is None:
        n_it = 200
    else:
        n_it = param.n_iter



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
        nzz = np.int( 0.5 * nz )

    if param.plot is True:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        Ln = ax1.imshow( x[nzz,i2:i1:-1,i1:i2] , animated=True , cmap=cm.Greys_r )
        plt.ion()



    ##  Start main loop
    while it < n_it and err > eps:
        print('\nSIRT --- iteraz. n.: ', it)

        #  --- step 0
        if it == 0:
            u = np.zeros( ( nz , n , n ) , dtype=myfloat )
            alpha = u.copy();  x_old = x.copy()


        ##  --- step 1
        x_old[:] = x.copy()
        pool = mproc.Pool()

        results = [ pool.apply_async( sirt_step , args=( x[i,:,:] , b[i,:,:] , tp , R , C ) ) \
                    for i in range( nz ) ]
        x = np.array( [ res.get() for res in results ] )
        pool.close()
        pool.join()


        if param.tv != 0.0:
            for i in range( nz ):
                x[i,i1:i2,i1:i2] = den.tv_breg( x[i,i1:i2,i1:i2] , param.tv )           


        if param.mask is not None:
            for i in range( nz ):
                x[i,i1:i2,i1:i2] = utils.supp_constr( x[i,i1:i2,i1:i2] , param.mask )

        if param.mask_add is not None:
            for i in range( nz ):
                aux = x[i,i1:i2,i1:i2].copy()
                for j in range( param.mask_add_n ):
                    aux[:] = utils.mask_constr( aux , param.mask_add[j] )
                x[i,i1:i2,i1:i2] = aux

        if param.pc >= 0.0:
            if param.pc == 1.0:  
                x[:,i1:i2,i1:i2] = utils.phys_constr( x[:,i1:i2,i1:i2] , 0.0 )
                x[:,i1:i2,i1:i2] = utils.resol_circle_constr( x[:,i1:i2,i1:i2] )
            if param.pc == 2.0:  
                x[:,i1:i2,i1:i2] = utils.resol_circle_constr( x[:,i1:i2,i1:i2] ) 
            else:
                x[:,i1:i2,i1:i2] = utils.phys_constr( x[:,i1:i2,i1:i2] , param.pc )     


        ##  Evaluate convergence
        diff  = np.linalg.norm( x - x_old ) / np.linalg.norm( x_old )
        obj = np.linalg.norm( tp.A( x[nzz,:,:] ) - b[nzz,:,:] )**2
        
        if param.checkit is True:
            if param.projector == 'grid-pswf' or param.projector == 'grid-kb':
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
        
        info.append( [ diff , obj ] ) 
            
        print('Relative difference: %.4e' % diff)
        print('Function score:      %.4e' % obj)
        
        import math
        if math.isnan( diff ) is True or math.isnan( obj ) is True:
            break


        ##  Plot intermediate reconstruction as check
        if param.plot is True:
            ax1.imshow( x[nzz,i2:i1:-1,i1:i2] , animated=True, cmap=cm.Greys_r )
            plt.draw()
            plt.pause(1)

        it +=1

    
    if param.plot is True:
        plt.ioff()

    return x , info




###########################################################
###########################################################
####                                                   #### 
####              ALGEBRAIC RECONSTRUCTION             ####
####                                                   ####
###########################################################
###########################################################

def alg_rec( b , a , param ):
    ##  Get number of pixels and angles
    m , n , nz = param.nang , param.npix_op , param.nz
    b  = np.array( b ).astype( myfloat )
    rd = param.radon_degree


    
    ##  Init forward and back-projector
    if param.projector == 'grid-pswf':
        tp = cpj1.projectors( n , a , kernel='pswf' , oversampl=2.0 , 
                              radon_degree=rd )
    elif param.projector == 'grid-kb':
        tp = cpj1.projectors( n , a , kernel='kb' , oversampl=1.5 , 
                             W=6.6, errs=6.0e-6, interp='lin' , radon_degree=rd )
    elif param.projector == 'bspline':
        tp = cpj2.projectors( n , a , param ,  bspline_degree=3 , 
                              proj_support_y=4 , radon_degree=rd )
    elif param.projector == 'slant':
        tp = cpj3.projectors( n , a , oper='ss' )
    elif param.projector == 'pix-driv':
        tp = cpj3.projectors( n , a , oper='pd' ) 
    elif param.projector == 'ray-driv':
        tp = cpj3.projectors( n , a , oper='rd' ) 
    elif param.projector == 'dist-driv':
        tp = cpj3.projectors( n , a , oper='dd' )


    
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



    ##  Reconstruction with SIRT
    if param.algorithm == 'sirt':
        x[:] , info = sirt( x , b , tp , param ) 

    
    
    ##  Conversion for bspline reconstruction and rotate
    if param.projector == 'bspline':
        for i in range( nz ):
            x[i,:,:] = bfun.convert_from_bspline_to_pixel_basis( x[i,:,:] , 3 )
        x[:,:,:] = x[:,::-1,::-1]

    
    return x , info  




##########################################################
##########################################################
####                                                  #### 
####               PLOT CONVERGENCE CURVES            ####
####                                                  ####
##########################################################
##########################################################

def plot_convergence_curves( info ):
    info = np.array( info ).reshape( len( info ) , 2 )

    fig , ax1 = plt.subplots()
    ax1.set_xlabel( 'Number of iterations' )
    ax1.set_ylabel( 'Arbitrary units' )   
    
    ax1.plot( info[:,0] , linewidth=3 , color='g' ,
              label='Relative difference' ); plt.hold( 'True' )
    ax2 = ax1.twinx()
    ax2.plot( info[:,1] , linewidth=3 , color='b' ,
              label='Objective score' ); plt.hold( 'True' )

    plt.suptitle( 'Convergence check plot' , fontsize=18 );
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
               ncol=2, fancybox=True, shadow=True)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 0.95),
               ncol=2, fancybox=True, shadow=True) 
    plt.show() 




##########################################################
##########################################################
####                                                  ####
####                SAVE RECONSTRUCTION               ####
####                                                  ####
##########################################################
##########################################################

def save_reco( pathout , filein , args , param , reco ):
    if args.reco is not None:
        filename = pathout + args.reco

    else:        
        filename = flein
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
    filename = args.filein
    filename = filename[:len(filename)-4] + param.root + '_logfile.txt' 

    fp = open( pathout + filename , 'w' )
    
    fp.write('ALGEBRAIC RECONSTRUCTION')
    
    today = datetime.date.today()
    
    fp.write('\n\nReconstruction performed on ' + str( today ) )
    fp.write('\n\nInput file:\n' + pathin + args.filein )
    fp.write('\n\nProjection angles:\n' + str( angles ))
    fp.write('\n\nCenter of rotation axis: ' + str( ctr ))
    
    if args.edge_padding is True:
        fp.write('\n\nEdge padding enabled')
    
    if args.dbp is True:
        fp.write('\n\nDifferential backprojection enabled')

    if args.n_iter is not None:
        fp.write('\n\nNumber of iterations:\n' + str( param.n_iter ))

    if args.eps is not None:          
        fp.write('\n\nStopping threshold:\n' + str( param.eps ))
    
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




###########################################################
###########################################################
####                                                   #### 
####                        MAIN                       ####
####                                                   ####
###########################################################
###########################################################

def main():
    ##  Initial print
    print('\n')
    print('##########################################################')   
    print('##########################################################')
    print('#####                                                #####')
    print('#####            Algebraic reconstruction            #####')
    print('#####                                                #####')     
    print('##########################################################')
    print('##########################################################')       
    print('\n')  



    ##  Get input arguments
    args = getArgs()



    ##  Get input and output paths
    pathin , pathout = utils.get_io_path( args )

    print('\nInput path:\n', pathin)
    print('\nOutput path:\n', pathout)    



    ##  Get input sinogram
    sino_list = [] ; filein = []
    
    if args.filein is not None:
        sinoname = pathin + args.filein  
        sino = io.readImage( sinoname ).astype( myfloat )
        if args.angle_pi is True:
            sino = sino[:sino.shape[0]-1,:]
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
    if nz == 1:
        nzz = 0
    else:
        nzz = np.int( 0.5 * nz )

    if args.plot is True:
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
    

    
    ##  Compute differential sinogram if DBP option enabled
    if args.dbp is True:
        print('\nComputing differential sinogram ....')
        for i in range( nz ):
            sino_list[i] = proc.diff_sino_savitzky_golay( sino_list[i] , window_size=args.sg ) 

        if args.plot is True:
            dis.plot( sino_list[0] , 'Differential sinogram' )
    


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


    
    ##  Getting stopping criterion
    print('\nSetup of the iterative procedure:')

    if nz == 0:
        labelout = pathout + filein[0][ : len( filein[0] ) - 4 ]
    else:
        labelout = pathout + filein[0][0][ : len( filein[0] ) - 4 ]     
    
    param = cap.art_param( npix_old , nang , nz , ctr , labelout , args )

    
    print( 'Projectors enabled: ' , args.projector )

    if args.dbp is True:
        print('DBP reconstruction enabled')
   
    if args.dpc is True:
        print('DPC reconstruction enabled')

    if args.n_iter is not None:
        print('Number of iterations: ', param.n_iter)

    if args.eps is not None:
        print('Stopping epsilon: ', param.eps)     

    if args.plot is True:
        print('Interactive plot:', param.plot)

    if args.logfile is True:
        print('Interactive plot:', param.logfile)  

    if args.init_object is True:
        print('\nInitialization with FBP reconstruction:', param.init_object)

    if param.mask is not None:
        print('\nObject support enabled')
        if param.plot is True:
            dis.plot( param.mask , 'Object support' )

    if param.mask_add is not None:
        print('\nAdditional supports provided')
        if param.plot is True:
            if param.mask_add_n == 1:
                dis.plot( param.mask_add[0] )
            else:
                dis.plot_multi( param.mask_add )

    if param.lt is True:
        print( '\nLocal tomography mode enabled' )



    ##  Iterative reconstruction
    print('\n\nAlgebraic reconstruction ....')
    time1 = time.time()
    reco_list , info = alg_rec( sino_list , angles , param )
    time2 = time.time()
    print('.... reconstruction done!')



    for i in range( nz ):
        ##  Crop reconstruction if edge-padding enabled
        if edf != 0.0:
            reco = reco_list[i,i1:i2,i1:i2]
        else:
            reco = reco_list[i,:,:] 


        ##  Show reconstruction
        if args.plot is True and i==0 and nz==1:
            dis.plot( reco , 'Reconstruction' )
            plot_convergence_curves( info )
        elif args.plot is True and nz==nzz:
            dis.plot( reco , 'Reconstruction' )
            plot_convergence_curves( info ) 

    
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
    print('#####        Algebraic reconstruction done!          #####')
    print('#####                                                #####')     
    print('##########################################################')
    print('##########################################################')       
    print('\n')   
    



###########################################################
###########################################################
####                                                   #### 
####                    CALL TO MAIN                   ####
####                                                   ####
###########################################################
########################################################### 

if __name__ == "__main__":
    main()
