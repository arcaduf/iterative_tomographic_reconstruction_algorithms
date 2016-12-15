##########################################################
##########################################################
####                                                  ####
####        CLASS TO HANDLE THE TOMOGRAPHIC           ####
####           FORWARD AND BACKPROJECTOR              ####
####                                                  ####
##########################################################
########################################################## 




####  PYTHON MODULES
from __future__ import division , print_function
import sys
import numpy as np




####  MY GRIDREC MODULE
cpath = '../common/'
sys.path.append( cpath + 'operators/gridrec_module/' );
sys.path.append( cpath + 'operators/gridrec_module/pymodule_gridrec_v4/' )
import gridrec_lut as glut
import gridrec_v4 as grid




####  MY FORMAT VARIABLES
myfloat = np.float32
myint = np.int




####  LIST OF FILTERS
filter_list = np.array( [ ['none',''] , ['ramp',''] , ['shepp-logan','shlo'] , 
                          ['hanning','hann'] , ['hamming','hamm']    ,
                          ['lanczos','lanc'] , ['parzen','parz' ]      ] )




####  CLASS PROJECTORS
class projectors:

    ##  Init class projectors
    def __init__( self , npix , angles , ctr=0.0 , kernel='pswf' , oversampl=2.0 , interp='nn' ,
                  radon_degree=0 , W=7.0 , errs=1e-3 , filt=None , args=None ):  
        
        ##  Get parameters from input arguments
        if args is not None:
            kernel       = args.kernel
            oversampl    = args.oversampl
            interp       = args.kernel_interp
            W            = args.kernel_size
            errs    = args.errs

            if hasattr( args , 'dpc' ) is True and args.dpc is True:
                radon_degree = 1
            else:
                radon_degree = 0

            if hasattr( args , 'dbp' ) is True and args.dbp is True:
                if args.dbp is True:
                    radon_degree = 1 
                else:
                    radon_degree = 0
            
            if hasattr( args , 'filt' ) is True:
                filt = args.filt
            else:
                filt = None

    
        
        ##  Compute regridding look-up-table and deapodizer
        W , lut , deapod = glut.configure_regridding( npix , kernel , oversampl , interp , W , errs )
    

        
        ##  Assign parameters
        if interp == 'nn':
            interp1 = 0
        else:
            interp1 = 1


        
        ##  Filter flag
        if filt is None:
            filt = 1
        else:
            filt = np.argwhere( filter_list == filt )[0][0]
            


        ##  Setting for forward nad backprojection
        param_nofilt = np.array( [ ctr , 0 , 0 , oversampl , interp1 , 
                                   len( lut ) - 5 , W , radon_degree ] )


        
        ##  Setting for filtered backprojection
        param_filt   = np.array( [ ctr , filt , 0 , oversampl , interp1 ,
                                   len( lut ) - 5 , W , radon_degree ] )  


        
        ##  Convert all input arrays to float 32
        self.lut          = lut.astype( myfloat )
        self.deapod       = deapod.astype( myfloat )
        self.angles       = angles.astype( myfloat )
        self.param_nofilt = param_nofilt.astype( myfloat )
        self.param_filt   = param_filt.astype( myfloat )


        
        ##  Overview of the projector settings
        print('\nSetting of the regridding projectors:')
        print('Kernel type: ', kernel)
        print('Oversampling: ', oversampl)
        print('Interpolation:', interp)
        print('LUT resolution: ', len( self.lut ))
        print('Kernel size: ' , W)
        print('Approx. error of LUT: ', errs)
        print('Filter for backprojection: ' , filter_list[filt,0])
        print('Radon transform degree: ', radon_degree)
    

    
    ##  Forward projector
    def A( self , x ):
        return grid.forwproj( x.astype( myfloat ) , self.angles , self.param_nofilt ,
                              self.lut , self.deapod )


    
    ##  Backprojector
    def At( self , x ):
        return grid.backproj( x.astype( myfloat ) , self.angles , self.param_nofilt , 
                              self.lut , self.deapod )


    
    ##  Filtered backprojection (with ramp filter as default)
    def fbp( self , x ):
        return grid.backproj( x , self.angles , self.param_filt ,
                              self.lut , self.deapod )  
