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
sys.path.append( cpath + 'operators/filters_module/' )  
import filters as fil
sys.path.append( cpath + 'operators/radon_module/pymodule_radon_pixel_driven' )   
sys.path.append( cpath + 'operators/radon_module/pymodule_radon_ray_driven' ) 
sys.path.append( cpath + 'operators/radon_module/pymodule_radon_distance_driven' ) 
sys.path.append( cpath + 'operators/radon_module/pymodule_radon_slant_stacking' )  
import radon_pixel_driven as rpd
import radon_ray_driven as rrd 
import radon_distance_driven as rdd
import radon_slant_stacking as rss




####  MY FORMAT VARIABLES
myfloat = np.float32
myint = np.int




####  CLASS PROJECTORS
class projectors:

    ##  Init class projectors
    def __init__( self , npix , angles , oper='pd' , ctr=0.0 ):
        nang = len( angles )
        angles1 = angles * np.pi / 180.0

        self.nang         = nang
        self.filt         = 'ramp'
        self.plot         = True
        self.angles       = angles1.astype( myfloat )
        self.oper         = oper


    
    ##  Forward projector
    def A( self , x ):
        if self.oper == 'pd':
            return rpd.forwproj( x.astype( myfloat ) , self.angles , 1 )
        elif self.oper == 'rd':
            return rrd.forwproj( x.astype( myfloat ) , self.angles )
        elif self.oper == 'dd':
            return rdd.forwproj( x.astype( myfloat ) , self.angles )
        elif self.oper == 'ss':
            return rss.forwproj( x.astype( myfloat ) , self.angles , 1 ) 


    
    ##  Backprojector
    def At( self , x ):
        if self.oper == 'pd':
            return rpd.backproj( x.astype( myfloat ) , self.angles , 1 )
        elif self.oper == 'rd':
            return rrd.backproj( x.astype( myfloat ) , self.angles )
        elif self.oper == 'dd':
            return rdd.backproj( x.astype( myfloat ) , self.angles )
        elif self.oper == 'ss':
            return rss.backproj( x.astype( myfloat ) , self.angles , 1 )  



    ##  Filtered backprojection
    def fbp( self , x ):
        ##  Option DPC
        x0 = x[:,::-1].copy()

        
        ##  Filtering projection
        x0[:] = fil.filter_proj( x0 , ftype=self.filt , dpc=dpc )
    
        
        ##  Backprojection
        if self.oper == 'pd':
            reco = rpd.backproj( x.astype( myfloat ) , self.angles , 1 )
        elif self.oper == 'rd':
            reco = rrd.backproj( x.astype( myfloat ) , self.angles )
        elif self.oper == 'dd':
            reco = rdd.backproj( x.astype( myfloat ) , self.angles )
        elif self.oper == 'ss':
            reco = rss.backproj( x.astype( myfloat ) , self.angles , 1 ) 


        ##  Normalization
        reco *= np.pi / ( 2.0 * self.nang )
        reco[:] = reco[::-1,::-1]

        return reco   
