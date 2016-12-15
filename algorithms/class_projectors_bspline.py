##########################################################
##########################################################
####                                                  ####
####        CLASS TO HANDLE THE TOMOGRAPHIC           ####
####           FORWARD AND BACKPROJECTOR              ####
####                                                  ####
##########################################################
########################################################## 




####  PYTHON MODULES
import sys
import numpy as np




####  MY GRIDREC MODULE
cpath = '../common/'   
sys.path.append( cpath + 'operators/bspline_module/' )
sys.path.append( cpath + 'operators/bspline_module/pymodule_genradon/' )
sys.path.append( cpath + 'operators/filters_module' )  
import bspline_functions as bfun
import genradon as gr
import my_image_display as dis
import filters as fil




####  MY FORMAT VARIABLES
myfloat = np.float32
myint = np.int




####  CLASS PROJECTORS
class projectors:

    ##  Init class projectors
    def __init__( self , npix , angles , param , ctr=0.0 , bspline_degree = 3 , proj_support_y=4 ,
                  nsamples_y=2048 , radon_degree=1 ):
    
        ##  Compute regridding look-up-table and deapodizer
        nang = len( angles )
        angles = np.arange( nang )
        angles = ( angles * 180.0 )/myfloat( nang )         
        lut = bfun.init_lut_bspline( nsamples_y , angles , bspline_degree ,
                                     radon_degree , proj_support_y )
        lut0 = bfun.init_lut_bspline( nsamples_y , angles , bspline_degree ,
                                      0 , proj_support_y ) 
        
       
        ##  Assign parameters
        self.nang         = nang
        self.filt         = 'ramp'
        self.plot         = True
        self.radon_degree = radon_degree
        self.lut          = lut.astype( myfloat )
        self.lut0         = lut0.astype( myfloat ) 
        self.angles       = angles.astype( myfloat )
        self.param_spline = np.array( [ lut.shape[1] , proj_support_y ] , dtype=myfloat )



    
    ##  Forward projector
    def A( self , x ):
        return gr.forwproj( x.astype( myfloat ) , self.angles , self.lut , self.param_spline )


    
    ##  Backprojector
    def At( self , x ):
        return gr.backproj( x.astype( myfloat ) , self.angles , self.lut , self.param_spline )



    ##  Filtered backprojection
    def fbp( self , x ):
        ##  Option DPC
        x0 = x[:,::-1].copy()
        if self.radon_degree == 0:
            dpc = False
        else:
            dpc = True

        
        ##  Filtering projection
        x0[:] = fil.filter_proj( x0 , ftype=self.filt , dpc=dpc )
    
        
        ##  Backprojection
        reco = gr.backproj( x0.astype( myfloat ) , self.angles , self.lut0 , self.param_spline )


        ##  Normalization
        if dpc is True:
            reco *= np.pi / ( 1.0 * self.nang )
        else:
            reco *= np.pi / ( 2.0 * self.nang )

        reco[:] = reco[::-1,::-1]

        return reco   

