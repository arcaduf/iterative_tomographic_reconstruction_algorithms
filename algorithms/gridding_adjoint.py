#################################################################################
#################################################################################
#################################################################################
#######                                                                   #######
#######                   SETUP REGRIDDING METHOD                         #######
#######                                                                   #######
#######        Author: Filippo Arcadu, arcusfil@gmail.com, 14/01/2013     #######
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
import multiprocessing as mproc    



####  MY PYTHON MODULES
cpath = '../common/'
sys.path.append( cpath + 'myimage/' )
import my_image_io as io
import my_image_display as dis
import my_image_process as proc
import utils




####  MY PROJECTOR CLASS
import class_projectors_grid as cpj 




####  MY FILTERING LIBRARY
sys.path.append( cpath + 'operators/filters_module/' )
import filters as fil  




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
    parser = argparse.ArgumentParser(description='Setup regridding backprojector',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-Di', dest='pathin', default='./',
                        help='Specify path to input data')    
    
    parser.add_argument('-i', dest='filein',
                        help='Specify name of input sinogram')

    parser.add_argument('-Do', dest='pathout',
                        help='Specify path to output data') 
    
    parser.add_argument('-o', dest='fileout',
                        help='Specify name of output reconstruction')
    
    parser.add_argument('-g', dest='geometry',default='0',
                        help='Specify projection geometry; @@@@@@@@@@@@@@@@@@@@@@@'
                             +' -g 0 --> equiangular projections between 0 and 180 degrees (default);'
                             +' @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@'
                             +' -g angles.txt use a list of angles (in degrees) saved in a text file')
    
    parser.add_argument('-f',dest='filt', default='parz',
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

    parser.add_argument('-z', dest='oversampl', type=myfloat, default=2,
                        help='Specify zero padding as fraction of the number of \
                             the pixels in the sinogram') 
    
    parser.add_argument('-c', dest='ctr', type=myfloat,
                        help='Centre of rotation (default: center of the image);'
                              + ' -1 ---> search for the center of rotation')

    parser.add_argument('-k', dest='kernel', default='prolate',
                        help='Choose window function: \n'
                            + '"cos" or "cosine"  --->  Cosine window\n'
                            + '"gauss" or "gaussian"  --->  Gaussian window\n'
                            + '"bsp" or "bspline"  --->  B-spline window\n'    
                            + '"kb" or "kaiser-bessel"  --->  Kaiser-Bessel window\n'
                            + '"pswf" or "prolate"  --->  Prolate spheroidal wavefunctions of degree 0\n')

    parser.add_argument('-kp', dest='param', type=myfloat,
                        help='Additional paramer for Gaussian , B-spline and'
                            +' Kaiser-Bessel interpolation kernels')

    parser.add_argument('-ks', dest='kernel_size', type=myfloat, default=7.0,
                        help='Specify kernel size; it corresponds to the "C"'
                            + ' constant in the original gridrec: thus, the kernel'
                            + ' radius corresponds to C/pi')

    parser.add_argument('-ki', dest='kernel_interp', default='nn',
                        help='Specify kernel interpolation type')  

    parser.add_argument('-ke', dest='errs', type=myfloat, default=1e-3,
                        help='Specify maximum tolerable aliasing error')

    parser.add_argument('-dpc', dest='dpc', action='store_true',
                        help='Enable DPC reconstruction')

    parser.add_argument('-dbp', dest='dbp', action='store_true',
                        help='Enable DBP reconstruction')

    parser.add_argument('-sg', dest='sg', type=myint, default=4,
                        help='Select Saviztky-Golay filter length')  

    parser.add_argument('-r', dest='resol', type=myint, default=1024,
                        help='Specify resolution of the interpolation LUT')     
                                                                                  
    parser.add_argument('-Z',dest='edge_pad', action='store_true',
                        help='Enable edge padding') 

    parser.add_argument('-p',dest='plot', action='store_true',
                        help='Display check-plots during the run of the code')

    parser.add_argument('-fo',dest='full_output', action='store_true',
                        help='Save output reconstruction with full output name')         
    
    args = parser.parse_args()
    
    return args




##########################################################
##########################################################
####                                                  ####
####               MULTI-THREAD FUNCTION              ####
####                                                  ####
##########################################################
##########################################################

def multi_thread( pathin , pathout , filein , args , ii ):
    sino = io.readImage( pathin + filein ).astype( myfloat )
    nang , npix = sino.shape
    angles = utils.create_projection_angles( nang )
    tp = cpj.projectors( npix , angles , args=args )    
    reco = tp.fbp( sino )
    if args.edge_pad:
        i1 = ii[0];  i2 = ii[2]
        reco = reco[i1:i2,i1:i2]
    save_reconstruction( pathout , filein , args , reco ) 




##########################################################
##########################################################
####                                                  ####
####                SAVE RECONSTRUCTION               ####
####                                                  ####
##########################################################
##########################################################

def save_reconstruction( pathout , filein , args , reco ):
    if args.fileout is None:
        filename = filein
        extension = filename[len(filename)-4:] 
        filename = filename[:len(filename)-4]

        if args.full_output is True:
            filename += '_' + args.kernel + '_' + args.kernel_interp
            oversampl = int( np.round( args.zero_pad * 1000 ) )
            filename += '_oversampl' + str( oversampl )

        if args.dbp is True:
            filename += '_dbp'

        filename += '_reco' + extension

    else:
        filename = args.fileout

    filename = pathout + filename

    print('\nWriting reconstruction in:\n', filename )    
    io.writeImage( filename , reco )




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
    print('########################################################')
    print('#############   REGRIDDING BACKPROJECTION  #############')
    print('########################################################')
    print('\n')


    ##  Get the startimg time of the reconstruction
    time1 = time.time()



    ##  Get input arguments
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



    ##  Getting projection geometry  
    if args.geometry == '0':
        print('\nDealing with equiangular projections distributed between 0 and'
                +' 180 degrees ---> [0,180)')
        angles = utils.create_projection_angles( nang )

    else:
        print('\nReading list of projection angles:\n', pathin + args.geometry)
        angles = utils.create_projection_angles( textfile = pathin + args.geometry ) 



    ##  Set center of rotation axis
    if args.ctr == None:
        ctr = 0.0
        print('Center of rotation axis placed at pixel: ', npix * 0.5)  
    else:
        if args.ctr == -1:
            ctr = proc.search_rot_ctr( sino , None , 'a' )
            print('Center of rotation axis placed at pixel: ', ctr)
        else:
            ctr = args.ctr
            print('Center of rotation axis placed at pixel: ', ctr)
            if args.dbp is True:
                sino[:,:] = proc.sino_correct_rot_axis( sino , ctr )
                print('Center of rotation corrected')
                ctr = 0.0



    ##  Enable edge padding
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



    ##  External sinogram filtering
    if args.filt == 'ramp-ext' or args.filt == 'conv-ext':
        if filt == 'ramp-ext':
            sino = fil.filter_fft( sino , 'ramp' )
        
        elif filt == 'conv-ext':
            sino = fil.filter_convolve( sino ) 

        sino *= 4.0 / myfloat( npix )
        args.filt = 0



    ##  Differential sinogram fo DBP
    if args.dbp is True:
        print( '\nDifferential backprojection enabled' )
        print( 'Computing differential sinogram by means of Savitzky-Golay method' )
        print( 'Savizky-Golay window length: ', args.sg )
        sino[:,:] = proc.diff_sino_savitzky_golay( sino , window_size=args.sg )

        if args.plot is True:
            dis.plot( sino , 'Differential sinogram' )



    ##  Initialize projectior class
    tp = cpj.projectors( npix , angles , ctr=ctr , args=args )     


    
    ##  Apply forward projection operator
    time_rec1 = time.time()
    reco = tp.fbp( sino )
    time_rec2 = time.time()  



    ##  Crop reconstruction
    if args.edge_pad:
        reco = reco[i1:i2,i1:i2]



    ##  Display reconstruction
    if args.plot is True:
        dis.plot( reco , 'Reconstruction' )


    
    ##  Save reconstruction
    save_reconstruction( pathout , file1 , args , reco )

    

    ##  Reconstruct sinograms from all the other images in the stack
    if args.filein is None:
        pool = mproc.Pool()
        for i in range( 1 , nimg ):
            pool.apply_async( multi_thread , ( pathin , pathout , file_list[0][i] , args , [i1,i2] ) )
        pool.close()
        pool.join()
    time2 = time.time()
    
    
    
    print('\nTime elapsed to run the 1st backward gridrec: ', time_rec2-time_rec1)
    print('Total time elapsed for the run of the program: ', time2-time1)


    
    print('\n')
    print('##############################################')
    print('####   REGRIDDING BACKPROJECTION DONE !   ####')
    print('##############################################')
    print('\n')




##########################################################
##########################################################
####                                                  ####
####               CALL TO MAIN                       ####
####                                                  ####
##########################################################
##########################################################  

if __name__ == '__main__':
    main()
