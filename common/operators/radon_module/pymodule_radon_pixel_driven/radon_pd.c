/*
 *  IMPLEMENTATION OF THE PIXEL-DRIVEN TOMOGRAPHIC PROJECTORS 
 */


#include <math.h>
#include <stdio.h>
#include <omp.h>

#define pi 3.141592653589793

float delta[ 8 ] = { 
                     -0.25 , -0.25 , 0.25 , -0.25 ,
                     -0.25 , 0.25 , 0.25 , 0.25  
                    };




void radon_pd( float* image , int npix , float *angles , int nang , int oper , int method , float *sino )
{
    int v, i, j, k, l, u, nh;
    float x0, y0, x, y, theta, s, c, t, uf, lf;

    nh = (int)( npix * 0.5 );

    int counter = 0;

    for( v=0 ; v<nang ; v++ ){
        theta = angles[v];
        s     = sin( theta );
        c     = cos( theta );

        for( i=0 ; i<npix ; i++ ){

            for( j=0 ; j<npix ; j++ ){
            
                x0 = j - (float)nh + 0.5;
                y0 = i - (float)nh + 0.5;

                for( k=0 ; k<4 ; k++ ){
                    x = x0 + delta[ 2*k + 1 ];
                    y = y0 + delta[ 2*k + 1 ];
                    t = fabs( x * s - y * c );

                    if( theta < pi/2 && y > x*s/c )
                        t = t;
                    else if( theta > pi/2 && y > x*s/c )
                        t = t;
                    else if( theta == 0 && y > 0 )
                        t = t;
                    else if( theta == pi/2 && x < 0 )
                        t = t;
                    else
                        t = -t;

                    if( method == 0 ){
                        l = (int)round( nh - 0.5 - t );
                        
                        if( oper == 0 )
                            sino[v*npix + l] += 0.25 * image[(npix-1-i)*npix + j];
                        else
                            image[(npix-1-i)*npix + j] += 0.25 * sino[v*npix + l];
                    }

                    else{
                        t  = nh - t;
                        lf = (float)floor( t );
                        uf = (float)ceil( t );
                        l  = (int)round( lf - 0.5 );
                        u  = (int)round( uf - 0.5 );
                        
                        if( oper == 0 ){
                            if( l > 0 && l < npix )
                                sino[v*npix + l] += 0.25 * image[(npix-1-i)*npix + j] * fabs( uf - t );
                            if( u > 0 && u < npix )                             
                                sino[v*npix + u] += 0.25 * image[(npix-1-i)*npix + j] * fabs( t - lf );
                        }
                        else{
                            if( l > 0 && l < npix )
                                image[(npix-1-i)*npix + j] += 0.25 * sino[v*npix + l] * fabs( uf - t );
                            if( u > 0 && u < npix )
                                image[(npix-1-i)*npix + j] += 0.25 * sino[v*npix + u] * fabs( t - lf );
                        }                        
                    }
                }        
            }        
        }
    }
}
