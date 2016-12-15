//This is a matrix form of the Radon Transform. Input lives in B-spline space (sample/image is expanded in Riesz basis),
//output is represented in 'canonical' basis in Radon space (i.e pixel basis, hence the sinogram can be made visible directly)
//compile
//gcc -O3 -fPIC -c generalized_radon_transform.c -o generalized_radon_transform.o
//make shared object
//gcc -shared -Wl -o generalized_radon_transform.so  generalized_radon_transform.o
//
// Remember:
// lut_size     --->   is supposed to be even
// half_pixel   --->   it may be set to 0.5, but for npix even, 0 is better



#include <math.h>
#include <stdio.h>
#include <omp.h>

#define pi 3.141592653589793



void gen_forwproj( float* image , int npix , float *angles , int nang , float *lut ,
                   int lut_size , float support_bspline , int num_cores , float *sino )
{
    float lut_step = ( lut_size * 0.5 ) / ( support_bspline * 0.5 );
    int lut_max_index = lut_size - 1;
    int lut_half_size = lut_size/2; 
    double half_pixel = 0.0; 
    double middle_right_det = 0.5 * npix + half_pixel; 
    double middle_left_det  = 0.5 * npix - half_pixel; 
    int delta_s_plus = (int)( 0.5 * support_bspline + 0.5 );

    int image_index , sino_index , lut_arg_index_c;
    
    float theta , COS , SIN , kx , ky , proj_shift , y , lut_arg_y;
    int theta_index , kx_index , ky_index , s , y_index, lut_arg_index;

    int chunk = ( int ) floor( nang / ( num_cores * 1.0 ) );

    #pragma omp parallel shared( image , npix , angles , nang , lut , lut_size , \
                                 support_bspline , sino , chunk , middle_left_det , \
                                 delta_s_plus , middle_right_det , lut_step , \
                                 half_pixel , lut_half_size ) \
                        private( theta_index , theta , COS , SIN , ky_index , ky , \
                                 kx_index , kx , proj_shift , image_index , s , \
                                 y_index , lut_arg_y , lut_arg_index , sino_index , \
                                 lut_arg_index_c )
    {
        #pragma omp for schedule( dynamic , chunk )
        for( theta_index = 0 ; theta_index < nang ; theta_index++ )
        {
            theta = angles[theta_index] * pi / 180.0;
            COS = cos(theta);
            SIN = sin(theta);
        
            for( ky_index = 0 ; ky_index < npix ; ky_index++ )
            {
                ky = -ky_index + middle_left_det;
            
                for ( kx_index = 0 ; kx_index < npix ; kx_index++ )
                {
                    kx = kx_index - middle_left_det;
                    proj_shift = COS * kx + SIN * ky;
                    image_index = ky_index * npix + kx_index;
                
                    for ( s = -delta_s_plus ; s <= delta_s_plus ; s++ )
                    {
                        y_index = (int)( proj_shift + s + middle_right_det );
                    
                        if (y_index >= npix || y_index < 0) continue;
            
                        y = y_index - middle_left_det;

                        lut_arg_y = y - proj_shift;
                        lut_arg_index = (int)( lut_arg_y * lut_step + half_pixel ) + lut_half_size;

                        if (lut_arg_index >= lut_max_index || lut_arg_index < 0) continue;

                        sino_index = theta_index * npix + y_index;
                        lut_arg_index_c = theta_index * lut_size + lut_arg_index;

                        sino[sino_index] += image[image_index] * lut[lut_arg_index_c]; 
                    }
                }
            }
        }
    }
}

