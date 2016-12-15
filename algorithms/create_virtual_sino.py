#################################################################################
#################################################################################
#################################################################################
#######                                                                   #######
#######                      CREATE VIRTUAL SINOGRAM                      #######
#######                                                                   #######
#######        Author: Filippo Arcadu, arcusfil@gmail.com, 22/02/2016     #######
#######                                                                   #######
#################################################################################
#################################################################################
#################################################################################




####  PYTHON MODULES
from __future__ import division , print_function
import time
import datetime
import argparse
import sys
import numpy as np
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk
from scipy.stats import norm




####  MY PYTHON MODULES
path_common = '../common/'
sys.path.append( path_common + 'pymodule_myimage/' )
import my_image_io as io
import my_image_display as dis
import my_image_process as proc
import utils




####  MY PROJECTOR CLASS
import class_projectors_grid as cpj 




####  MY FORMAT VARIABLES & CONSTANTS
myfloat = np.float32
myint = np.int
pi = np.pi
eps = 1e-8 




##########################################################
##########################################################
####                                                  ####
####             GET INPUT ARGUMENTS                  ####
####                                                  ####
##########################################################
##########################################################

def getArgs():
    parser = argparse.ArgumentParser(description='Create virtual sinogram',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-Di', dest='pathin', default='./',
                        help='Specify path to input data')    
    
    parser.add_argument('-i', dest='filein',
                        help='Specify name of input sinogram')

    parser.add_argument('-Do', dest='pathout',
                        help='Specify path to output data') 
    
    parser.add_argument('-o', dest='fileout',
                        help='Specify name of output reconstruction')
    
    parser.add_argument('-f',dest='filt', default='hamm',
                        help = 'Specify filter: ' 
                        + 'none  --->  non-filtered backprojection'
                        + 'ramp  --->  ramp or Ram-Lak filter'
                        + 'shlo  --->  Shepp-Logan filter'
                        + 'hann  --->  Hanning filter'
                        + 'hamm  --->  Hamming filter'
                        + 'parz  --->  Parzen filter'
                        + 'lanc  --->  Lanczos filter'
                        + 'ramp-ext  --->  Ramp external'
                        + 'conv-ext  --->  Convolution external') 

    parser.add_argument('-c', dest='ctr', type=myfloat,
                        help='Centre of rotation (default: center of the image);'
                              + ' -1 ---> search for the center of rotation')

    parser.add_argument('-e', dest='edge_pad', action='store_true',  
                        help='Enable edge-padding')   

    parser.add_argument('-r', dest='back_rem', action='store_true',
                        help='Background equalization')      

    parser.add_argument('-p',dest='plot', action='store_true',
                        help='Display check-plots during the run of the code')

    args = parser.parse_args()
    
    return args




##########################################################
##########################################################
####                                                  ####
####               BACKGROUND EQUALIZATION            ####
####                                                  ####
##########################################################
##########################################################

def background_equalization( img , size=20 ):
    nx , ny = img.shape
    tracer  = np.min( img ) - 100
    se      = disk( size )
    yy , xx = np.meshgrid( np.arange( ny ) , np.arange( nx ) )

    
    img_e   = erosion( img , se )
    img_e = img_e.reshape( -1 )


    mu , std = norm.fit( img_e )
    img_e[ ( img_e < mu - 0.3 * std ) | ( img_e > mu + 0.3 * std )  ] = tracer
    it = np.argwhere( img_e != tracer ).reshape( -1 )


    X = np.hstack( ( np.ones( ( nx * ny , 1 ) ) ,
                     xx.reshape( nx * ny , 1 ) ,
                     yy.reshape( nx * ny , 1  ) ) )
    X1 = X[it,:]
    M  = np.linalg.lstsq( X1 , img_e[it] )[0].reshape( 3 , 1 )
    back = np.array( np.dot( X , M ) ).reshape( nx , ny )
    new = img - back
    new = ( new - np.min( new ) ) * 255 / ( np.max( new ) - np.min( new ) )
    new = new.astype( np.uint8 )

    return new



##########################################################
##########################################################
####                                                  ####
####                         MAIN                     ####
####                                                  ####
##########################################################
##########################################################

def main():
    ##  Initial print
    print('\n')
    print('########################################################')
    print('####              CREATE VIRTUAL SINOGRAM           ####')
    print('########################################################')
    print('\n')  



    ##  Get arguments
    args = getArgs()



    ##  Get input/output directory
    pathin , pathout = utils.get_io_path( args )
    
    print('\nInput path:\n', pathin)
    print('\nOutput path:\n', pathout)



    ##  Get input files
    file_list , file1 , nimg , ext = utils.get_input( args , pathin )
    
    print('\nNumber of input files: ' , nimg)
    print('Extension of the files: ', ext)



    ##  Read first sinogram
    sino = io.readImage( pathin + file1 ).astype( myfloat )
    nang , npix = sino.shape

    print('\nSinogram to reconstruct:\n', file1)
    print('Number of projection angles: ', nang)
    print('Number of pixels: ', npix)     

    if args.plot is True:
        dis.plot( sino , 'Input sinogram' )



    ##  Set center of rotation axis
    if args.ctr == None:
        ctr = 0.0
        print('Center of rotation axis placed at pixel: ', npix * 0.5)  
    else:
        ctr = args.ctr
        print('Center of rotation axis placed at pixel: ', ctr)



    ##  Enable edge-padding
    if args.edge_pad is True:
        sino = proc.sino_edge_padding( sino , 0.5 ).astype( myfloat )
        i1 = int( ( sino.shape[1] - npix ) * 0.5 )
        i2 = i1 + npix            
        npix = sino.shape[1]

        if ctr != 0.0:
            ctr += i1
        
        if args.plot is True:
            dis.plot( sino , 'Edge padded sinogram' )
    else:
        i1 = 0
        i2 = npix 



    ##  Prepare projectors
    ang = np.arange( nang ) * 180.0 / myfloat( nang )  
    tp = cpj.projectors( npix , ang , kernel='kb' , oversampl=2.32 , 
                         W=6.6 , errs=6.0e-6 , interp='lin' , 
                         radon_degree=0 , filt=args.filt , ctr=ctr ) 



    ##  Reconstruct
    reco = tp.fbp( sino )
    if args.plot is True:
        dis.plot( reco , 'Reconstruction' ) 
    #reco = reco[i1:i2,i1:i2]



    ##  Zero-out pixels outside resolution circle
    reco_new = reco.copy();  reco_new[:] = 0.0
    io.writeImage( 'reco.DMP' , reco[i1:i2,i1:i2] ) 
    reco_new[i1:i2,i1:i2] = utils.resol_circle_constr( reco[i1:i2,i1:i2] )
    reco[:] = reco_new[:]

    if args.plot is True:
        dis.plot( reco , 'Constrained reconstruction' )
    io.writeImage( 'reco_circle.DMP' , reco )



    ##  Background equalization
    reco[:] = background_equalization( reco )

    if args.plot is True:
        dis.plot( reco , 'Equalized reconstruction' )
    io.writeImage( 'reco_equaliz.DMP' , reco )



    ##  Forward projection
    nang_new = np.int( npix * np.pi / 2.0 )
    ang_new  = np.arange( nang_new ) * 180.0 / np.float32( nang_new )
    tp       = cpj.projectors( npix , ang_new , kernel='kb' , oversampl=2.32 , 
                               W=6.6 , errs=6.0e-6 , interp='lin' , 
                               radon_degree=0 ) 
    sino = tp.A( reco )
    #sino = sino[:,i1:i2]

    if args.plot is True:
        dis.plot( sino , 'Forward projection' )
    io.writeImage( 'sino_circle.DMP' , sino )



    ##  Save output file
    if args.fileout is None:
        filein    = args.filein
        extension = filein[len(filein)-4:]
        fileout   = filein[:len(filein)-4] + '_virt.tif'
    else:
        fileout = args.fileout
    io.writeImage( pathout + fileout , sino )
    print( '\nWritten output file:\n' , pathout + fileout )




##########################################################
##########################################################
####                                                  ####
####                         MAIN                     ####
####                                                  ####
##########################################################
##########################################################

if __name__ == '__main__':
    main()
