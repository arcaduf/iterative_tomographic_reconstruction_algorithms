from __future__ import division , print_function
import numpy as np
import sys
import os
import shutil

myint = np.int


class sir_param:
    def __init__( self , nang , npix , nz , ctr , labelout , args ):
        ##  Entries with no check
        self.ctr           = ctr
        self.nang          = nang
        self.nz            = nz
        self.eps           = args.eps
        self.n_iter        = args.n_iter
        self.plot          = args.plot
        self.logfile       = args.logfile
        self.reg_cost      = args.reg_cost
        self.huber_cost    = args.huber_cost
        self.init_object   = args.init_object
        self.algorithm     = args.algorithm
        self.num_cores     = args.num_cores


        ##  Tomographic projectors
        self.projector = args.projector


        ##  Handling edge padding
        if args.lt is True:
            pad_factor = 0.87
        elif args.lt is False and args.edge_padding:
            pad_factor = args.edge_padding
        else:
            pad_factor = 0.0
            
        if pad_factor == 0:
            self.index_start = 0
            self.index_end = npix
            self.edge_padding = False
            self.npix_op = npix
        else:
            npad = int( pad_factor * npix ) 
            npix_new = npix + 2 * npad
            self.index_start = myint( 0.5 * ( npix_new - npix ) )
            self.index_end = self.index_start + npix
            self.edge_padding = True
            self.npix_op = npix_new 


        ##  Handling regularization
        self.reg = None
        reg = args.regularization
        if reg is not None:
            if reg == 'h' or reg == 'huber':
                self.reg = 'huber'
            elif reg == 't' or reg == 'tikhonov':
                self.reg = 'tikhonov'
            elif reg == 'a' or reg == 'haar':
                self.reg = 'haar'
            else:
                print( 'Warning regularization "', reg,'" is not available!' )
                print( 'Reconstructing with no regularization' )


        ##  Output name root
        if args.algorithm == 'sps':
            root = '_sps'
        elif args.algorithm == 'em':
            root = '_em'

        if self.reg == 'huber':
            root += '_huber'
        elif self.reg == 'tikhonov':
            root += '_tikh'
        elif self.reg == 'haar':
            root += '_haar'

        if self.projector == 'grid-kb':
            root += '_grid_kb'
        elif self.projector == 'grid-pswf':
            root += '_grid_pswf'
        elif self.projector == 'bspline':
            root += '_bspline'
        elif self.projector == 'radon':
            root += '_radon'
        elif self.projector == 'pix-driv':
            root += '_pd'  
        elif self.projector == 'ray-driv':
            root += '_rd'  
        elif self.projector == 'dist-driv':
            root += '_dd'              

        self.root = root 


        ##  Handling stopping criterion
        if self.eps is None:
            self.eps = -1.0
        elif self.n_iter is None:
            self.n_iter = 1e20

        
        ##  Saving each iteration result for analysis
        if args.checkit is not None:
            self.checkit = True
            path_rmse = '_rmse_folder'
            path_rmse = args.checkit + path_rmse + '/'
            if not os.path.exists( path_rmse ):
                os.makedirs( path_rmse )
            else:
                shutil.rmtree( path_rmse )
                os.makedirs( path_rmse )
            self.path_rmse = path_rmse
        else:
            self.checkit = False        
