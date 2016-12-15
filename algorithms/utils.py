########################################################################
########################################################################
####                                                                #### 
####             Utilities for iterative reconstruction             ####
####                                                                ####
####     Author: Filippo Arcadu, arcusfil@gmail.com, 12/02/2015     ####  
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




####  MY VARIABLE TYPE
myint   = np.int
myfloat = np.float32




#############################################################
#############################################################
####                                                     ####
####               HANDLING INPUT ARGUMENTS              ####
####                                                     ####
#############################################################
#############################################################

##  Deal with input and output path
def get_io_path( args ):
    pathin = args.pathin
    if pathin[len(pathin)-1] != '/':
        pathin += '/'

    if args.pathout is None:
        pathout = pathin
    else:
        pathout = args.pathout
        if pathout[len(pathout)-1] != '/':
            pathout += '/'
        if os.path.exists( pathout ) is True and args.filein is None:
            shutil.rmtree( pathout )
            os.makedirs( pathout )
        elif os.path.exists( pathout ) is False:
            os.makedirs( pathout )

    return pathin , pathout



##  Deal with the input data (single file or stack)
def get_input( args , pathin ):
    file_list  = []
    if args.filein is not None:
        file1 = args.filein 
        print('\nReading single input file:\n', file1)
        ext = file1[len(file1)-4:]
        nimg  = 1

    else:
        print('\nReading stack of images placed\n' )
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

        file_list.append( sorted( glob.glob( '*' + ext ) ) )
        nimg = len( file_list[0] )
        file1 = file_list[0][0]
        os.chdir( curr_dir )

    return file_list , file1 , nimg , ext




#############################################################
#############################################################
####                                                     ####
####               CREATE PROJECTION ANGLES              ####
####                                                     ####
#############################################################
#############################################################

def create_equally_spaced_angles( nang , angle_start , angle_end ):
    angles = np.linspace( angle_start , angle_end , nang , endpoint=False )
    return angles  



def create_pseudo_polar_angles( nang ):
    if nang % 4 != 0:
        raise Exception('\n\tError inside createPseudoPolarAngles:'
                        +'\n\t  nang (input) is not divisible by 4 !\n')
    n = nang
    pseudo_angles = np.zeros(n,dtype=myfloat)
    pseudoAlphaArr = np.zeros(n,dtype=myfloat)
    pseudoGridIndex = np.zeros(n,dtype=int)
    index = np.arange(int(n/4)+1,dtype=int)

    pseudo_angles[0:int(n/4)+1] = np.arctan(4*index[:]/myfloat(n))
    pseudo_angles[int(n/2):int(n/4):-1] = np.pi/2-pseudo_angles[0:int(n/4)]    
    pseudo_angles[int(n/2)+1:] = np.pi-pseudo_angles[int(n/2)-1:0:-1]

    pseudo_angles *= 180.0 / np.pi
    
    return pseudo_angles



##  Note: this function gives as output angles in degrees,
##        excluding the right extreme, e.g., [ 0.0 , 180 )

def create_projection_angles( nang=None , start=0.0 , end=180.0 , 
                              pseudo=0 , wedge=False , textfile=None ):
    
    ##  Create angles
    if textfile is None:
        ##  Create equally angularly space angles in [alpha_{0},alpha_{1})
        if pseudo == 0:
            if wedge is False:
                angles = create_equally_spaced_angles( nang , start , end )

            else:
                angles1 = create_equally_spaced_angles( nang , 0.0 , start )
                angles2 = create_equally_spaced_angles( nang , end , 180.0 )
                angles = np.concatenate( ( angles1 , angles2 ) , axis=1 )

        ##  Create equally sloped angles
        else:
            angles = create_pseudo_polar_angles( nang )


    ##  Read angles from text file
    else:
        angles = np.fromfile( textfile , sep="\t" ) 


    return angles

    


#############################################################
#############################################################
####                                                     ####
####              APPLY PHYSICAL CONSTRAINTS             ####
####                                                     ####
#############################################################
#############################################################

def phys_constr( x , beta = 0.9 ):
    i = np.argwhere( x < 0 )
    
    x[i[:,0],i[:,1],i[:,2]] *= beta
    
    return x




#############################################################
#############################################################
####                                                     ####
####                   SOFT THRESHOLDING                 ####
####                                                     ####
#############################################################
#############################################################

def soft_thres( x , beta ):
    if beta <=0:
        raise Error( 'beta in soft thresholding has to be > 0' )

    i1 = np.argwhere( x > beta )    
    i2 = np.argwhere( np.abs( x ) <= beta )
    i3 = np.argwhere( x < -beta ) 
    
    x[i1[:,0],i1[:,1],i1[:,2]] -= beta
    x[i2[:,0],i2[:,1],i2[:,2]] = 0
    x[i3[:,0],i3[:,1],i3[:,2]] += beta        
    
    return x




#############################################################
#############################################################
####                                                     ####
####                   SOFT THRESHOLDING                 ####
####                                                     ####
#############################################################
#############################################################

def hard_thres( x , beta ):
    t = np.maximum( 0 , x - beta ) - np.maximum( 0 , -x -beta ) 
    return t




#############################################################
#############################################################
####                                                     ####
####    ZERO-OUT REGIONS OUT OF THE RESOLUTION CIRCLE    ####
####                                                     ####
#############################################################
#############################################################

def resol_circle_constr( img ):
    if img.ndim == 2:
        m , n = img.shape
    else:
        nz , m , n = img.shape

    ctr_x = m * 0.5 - 0.5
    ctr_y = n * 0.5 - 0.5

    d = np.min( ( ctr_x , ctr_y ) )

    x = np.arange( m )
    y = np.arange( n )
    x , y = np.meshgrid( x , y )

    i = np.argwhere( np.sqrt( ( x - ctr_x )**2 + ( y - ctr_y )**2 ) > d )

    if img.ndim == 2:
        img[i[:,0],i[:,1]] = 0
    else:
        img[:,i[:,0],i[:,1]] = 0
    
    return img




#############################################################
#############################################################
####                                                     ####
####                    SUPPORT CONSTRAINT               ####
####                                                     ####
#############################################################
############################################################# 

def supp_constr( x , mask ):
    x[ mask == 0 ] = 0.0
    return x




#############################################################
#############################################################
####                                                     ####
####                      MASK CONSTRAINT                ####
####                                                     ####
#############################################################
############################################################# 

def mask_constr( x , mask ):
    x[ mask == 1 ] = np.mean( x[ mask == 1 ] )
    return x
