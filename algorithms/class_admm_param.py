from __future__ import division , print_function
import numpy as np
import sys
import os
import shutil
sys.path.append( '../common/myimage/' )
import my_image_io as io


myint   = np.int
myfloat = np.float32


class admm_param:
    def __init__( self , npix , nang , nz , ctr , labelout , args ):
        ##  Entries with no check
        self.ctr           = ctr
        self.nang          = nang
        self.nz            = nz
        self.n_iter        = args.n_iter
        self.n_iter_pcg    = args.n_iter_pcg
        self.eps            = args.eps
        self.plot          = args.plot
        self.logfile       = args.logfile
        self.init_object   = args.init_object
        self.pc            = args.pc   
        self.mask          = []
        self.lt            = args.lt
        self.num_cores     = args.num_cores


        ##  Tomographic projectors
        self.projector = args.projector


        ##  Enable DPC reconstruction
        if args.dpc is True or args.dbp is True:
            self.radon_degree  = 1
        else:
            self.radon_degree  = 0


        ##  Check point
        if self.radon_degree == 1:
            if self.projector == 'pix-driv' or self.projector == 'ray-driv' \
               or self.projector == 'dist-driv':
                sys.exit( '\nERROR: selected projector ' + self.projector + \
                          ' cannot be used for differentiated tomography!!\n')    


        ##  Handling regularization
        if args.reg is not None:
            self.reg = args.reg

        if args.param is None:
            if self.reg == 'lasso' or self.reg == 'lasso-tv':
                param = '1.0:1.0:1.0'
            elif self.reg == 'pp-breg':
                param = '1.0:100.0:1.0' 
        elif self.reg != 'cg':
            param = args.param


        ##  Handling parameters
        if self.reg != 'cg': 
            param = param.split( ':' )
            self.lambd1 = myfloat( param[0] )
            self.lambd2 = myfloat( param[1] )
            self.mu     = myfloat( param[2] )
            self.relax  = 1.0
            self.z1     = args.inner_padding

        
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


        ##  Output name root
        if self.reg != 'cg':
            root = '_admm'

            if self.reg == 'lasso':
                root += '_lasso'
            elif self.reg == 'lasso-tv':
                root += '_lassotv'
            elif self.reg == 'pp-breg':
                root += '_ppbreg'
            elif args.reg == 'pp-rof':
                root += '_pprof'
            elif args.reg == 'pp-breg':
                root += '_ppbreg'
            elif args.reg == 'pp-chamb':
                root += '_ppchamb'
            elif args.reg == 'pp-nlmeans':
                root += '_ppnlmeans'         
            elif args.reg == 'pp-tgv':
                root += '_pptgv'
            elif args.reg == 'pp-nltv':
                root += '_ppnltv'             

            string = '_lambd1_' + str( self.lambd1 ) + \
                     '_lambd2_' + str( self.lambd2 ) + \
                     '_mu_' + str( self.mu )
            root += string
        else:
            root = '_cgls'

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


        ##  Object support
        if args.mask is not None:
            self.mask = io.readImage( args.mask ).astype( np.uint8 )
        else:
            self.mask = None


        ##  Additional masks
        if args.mask_add is not None:
            if args.mask_add.find( ',' ) != -1:
                files  = args.mask_add.split( ',' )
                nfiles = len( files )
                self.mask_add = []
                for i in range( nfiles ):
                    self.mask_add.append( io.readImage( files[i] ).astype( np.uint8 ) )
                self.mask_add_n = nfiles   
            else:
                files  = args.mask_add
                nfiles = 1  
                self.mask_add = []
                self.mask_add.append( io.readImage( files ).astype( np.uint8 ) )
                self.mask_add_n = nfiles
        else:
            self.mask_add = None
