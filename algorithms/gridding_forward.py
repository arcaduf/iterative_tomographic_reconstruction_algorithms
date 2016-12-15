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
import my_print as pp
import my_image_display as dis
import utils 

            


####  MY PROJECTOR CLASS
import class_projectors_grid as cpj




####  MY FORMAT VARIABLES
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
    parser = argparse.ArgumentParser(description='Setup forward regridding projector',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-Di', dest='pathin', default='./',
                        help='Specify path to input image')    
    
    parser.add_argument('-i', dest='filein',
                        help='Specify name of input sinogram')

    parser.add_argument('-Do', dest='pathout',
                        help='Specify path to output data') 
    
    parser.add_argument('-o', dest='fileout',
                        help='Specify name of output sinogram')
    
    parser.add_argument('-n', dest='nang', type=np.int,
                        help='Specify number of projection angles')    
    
    parser.add_argument('-g', dest='geometry', default='0',
                        help='Specify projection geometry;'
                             +' -g 0 --> equiangular projections between 0 and'
                             +' 180 degrees (default); -g 1 --> equally sloped'
                             +' projections; -g angles.txt use a list of angles'
                             +' (in degrees) saved in a text file')

    parser.add_argument('-an', dest='angle_range', default='0',
                        help='Specify angle range: -a 23.5 ---> angles in '
                             + '[23.5,23.5+180); -a 35.0:135.0 ---> angles in '
                             + '[35.0,135.0)') 

    parser.add_argument('-z', dest='oversampl', type=myfloat, default=2,
                        help='Specify zero padding as fraction of the number of \
                             the pixels in the sinogram')      
    
    parser.add_argument('-c', dest='centre',default=-1,type=myfloat,
                        help='Centre of rotation (default: centre of the image);'
                              + ' if working with more images a list of rotation centre'
                              + ' can be given to this parameter')

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

    parser.add_argument('-r', dest='resol', type=myint, default=1024,
                        help='Specify resolution of the interpolation LUT')

    parser.add_argument('-dpc', dest='dpc', action='store_true',
                        help='Enable DPC projector') 

    parser.add_argument('-p', dest='plot', action='store_true',
                        help='Display check-plots during the run of the code')
    
    parser.add_argument('-l',dest='list_ang', action='store_true',
                        help='Write the list of projection angles')

    parser.add_argument('-fo',dest='full_output', action='store_true',
                        help='Save output reconstruction with full output name')      
    
    args = parser.parse_args()
    
    if args.nang is None and args.geometry in ['0','1']:
        parser.print_help()
        sys.exit('\nERROR: Input number of angles not specified!\n')    
    
    return args




##########################################################
##########################################################
####                                                  ####
####               MULTI-THREAD FUNCTION              ####
####                                                  ####
##########################################################
##########################################################

def multi_thread( pathin , pathout , filein , angles , args ):
    image = io.readImage( pathin + filein ).astype( myfloat )
    npix = image.shape[0]
    tp = cpj.projectors( npix , angles , args=args ) 
    sino = tp.A( image )  
    save_sinogram( pathout , filein , angles , args , sino )




##########################################################
##########################################################
####                                                  ####
####                    SAVE SINOGRAM                 ####
####                                                  ####
##########################################################
##########################################################

def save_sinogram( pathout , filein , angles , args , sino ):
    nang = len( angles )

    if args.fileout is None or args.filein is None:
        filename = filein
        extension = filename[len(filename)-4:] 
        filename = filename[:len(filename)-4]

        nang = args.nang
        if nang < 10:
            str_nang = '000' + str( nang )
        elif nang < 100:
            str_nang = '00' + str( nang )
        elif nang < 1000:
            str_nang = '0' + str( nang )    
        else:
            str_nang = str( nang )

        filename += '_ang' + str_nang

        if args.dpc is True:
            filename += '_dpc'

        if args.full_output is True:
            if args.geometry == '0':
                filename += '_polar'
            elif args.geometry == '1':
                filename += '_pseudo'
            else:
                filename += '_text'
            filename += '_' + args.kernel + '_' + args.kernel_interp
            oversampl = int( np.round( args.zero_pad * 1000 ) )
            filename += '_oversampl' + str( oversampl )        
        filename += '_sino' + extension

    else:
        filename = args.fileout

    filename = pathout + filename

    print('\nWriting sinogram in:\n', filename )    
    io.writeImage( filename , sino )


    ##  Save list of projection angles
    if args.list_ang is True:
        listang = filename[:len(filename)-4]
        listang += '_ang' + str( nang ) + '_list.txt'

        print('\nWriting list of projection angles:\n', listang) 
        fd = open( listang , 'w' )
        for i in range( len( angles ) ):
            fd.write('%.8f\n' % angles[i] )
        fd.close()     




##########################################################
##########################################################
####                                                  ####
####                       MAIN                       ####
####                                                  ####
##########################################################
##########################################################

#@profile
def main():
    ##  Initial print
    print('\n')
    print('###########################################################')
    print('#############   FORWARD REGRIDDING PROJECTOR  #############')
    print('###########################################################')
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



    ##  Read first image
    image = io.readImage( pathin + file1 ).astype( myfloat )
    npix = image.shape[0]

    print('\nFirst image to forward project: ', file1)
    print('Number of pixels: ', npix)

    if args.plot is True:
        dis.plot( image , 'Input image' )



    ##  Get projection angles
    if args.geometry == '0' or args.geometry == '1':
        nang       = args.nang
        angle_type = myint( args.geometry )

        if args.angle_range.find( ':' ) == -1 or args.geometry == -1:
            angle_start = myfloat( args.angle_range )
            angle_end = angle_start + 180.0

        else:
            angle_aux = args.angle_range.find( ':' )
            angle_start = myfloat( angle_aux[0] )
            angle_end = myfloat( angle_aux[1] )

        angles = utils.create_projection_angles( nang , angle_start , angle_end )

    else:
        angles = utils.create_projection_angles( pathin + args.geometry ) 
    
    nang = len( angles )

    print('\nNumber of views: ', nang)
    print('Selected angle range: [ ', angle_start,' , ', angle_end,' )' )
    print('Angles:\n', angles)
    


    ##  Initialize projectior class
    tp = cpj.projectors( npix , angles , args=args ) 


    
    ##  Apply forward projection operator
    time_rec1 = time.time()
    sino = tp.A( image )
    time_rec2 = time.time()

    

    ##  Display sinogram     
    if args.plot is True:
        dis.plot( sino , 'Sinogram' )



    ##  Save sinogram
    save_sinogram( pathout , file1 , angles , args , sino )

    
    
    ##  Create sinograms from all the other images in the stack
    if args.filein is None:
        pool = mproc.Pool()
        for i in range( 1 , nimg ):
            pool.apply_async( multi_thread , ( pathin , pathout , file_list[0][i] , angles , args ) )
        pool.close()
        pool.join()
    time2 = time.time()
    
    
    
    print('\nTime elapsed to run the 1st forward gridrec: ', time_rec2-time_rec1)
    print('Total time elapsed for the run of the program: ', time2-time1)


    
    print('\n')
    print('#######################################')
    print('####    FORWARD PROJECTION DONE !  ####')
    print('#######################################')
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
