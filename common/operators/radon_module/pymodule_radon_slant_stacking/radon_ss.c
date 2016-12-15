/*
 *  IMPLEMENTATION OF THE PIXEL-DRIVEN TOMOGRAPHIC PROJECTORS 
 */


#include <math.h>
#include <stdio.h>
#include <omp.h>

#define pi 3.141592653589793
#define eps 1e-5
#define s45 0.70710678118654757



void quick_sort ( float *a , int n ) {
    int i, j;
    float p, t;

    if ( n < 2 )
        return;

    p = a[ n / 2 ];
    
    for ( i = 0 , j = n - 1 ;; i++ , j-- ){
        while ( a[i] < p )
            i++;
        while ( p < a[j] )
            j--;
        if ( i >= j )
            break;
        t = a[i];
        a[i] = a[j];
        a[j] = t;
    }
    quick_sort( a , i );
    quick_sort( a + i , n - i );
}




void ray_limits( float* lim, int t , float s , float c , int npix  ){
    float x1, y1, x2, y2;
    float x_min, x_max, y_min, y_max;

    int nh = (int)( npix * 0.5 );

    
    //  Theta = 0
    if( fabs( s ) < eps ){
        lim[0] = 0;
        lim[1] = 0;
        lim[2] = -nh;
        lim[3] = nh-1;
    }


    //  Theta = pi/2
    else if( fabs( c ) < eps ){
        lim[0] = -nh;
        lim[1] = nh-1;
        lim[2] = 0;
        lim[3] = 0;      
    
    }


    //  All other angles
    else{
        x1 = t/c + nh * s/c;
        y1 = t/s + nh * c/s;
        x2 = t/c - (nh-1) * s/c;
        y2 = t/s - (nh-1) * c/s;

        lim[0] = x1;  lim[1] = x2;  lim[2] = -nh;  lim[3] = nh-1;
        quick_sort( lim , 4 );
        x_min = lim[1];
        x_max = lim[2];

        lim[0] = y1;  lim[1] = y2;  lim[2] = -nh;  lim[3] = nh-1;
        quick_sort( lim , 4 );
        y_min = lim[1];
        y_max = lim[2];

        lim[0] = x_min;  lim[1] = x_max;  lim[2] = y_min;  lim[3] = y_max; 
    }
}




void radon_ss( float* image , int npix , float *angles , int nang , int oper , int method , float *sino )
{
    int v, j, k, t, nh, x1, y1, x, y;
    float theta, s, c, w, xf, yf;
    
    float *lim = ( float* )calloc( 4 , sizeof( int ) );

    nh = (int)( npix * 0.5 );


    for( v=0 ; v<nang ; v++ ){
        theta = angles[v];
        s     = sin( theta );
        c     = cos( theta );
            
        
        for( t=-nh ; t<nh ; t++ ){
            //  Indeces for sinogram
            j = nang - 1 - v;
            k = t + nh;


            //  Find ray limits
            ray_limits( lim , t , s , c , npix );

            
            //  0 <= theta <= pi/4  U  3pi/4 < theta <= pi
            if( fabs( s ) <= s45 ){
                for( y=(int)round(lim[2]) ; y<(int)round(lim[3]) ; y++ ){
                    //  Nearest neighbour interpolation
                    if( method == 0){
                        x1 = (int)round( t/c - y*s/c ) + nh;
                        y1 = y + nh;

                        if( oper == 0 )
                            sino[ j * npix + k ] += 1.0/fabs(c)  * image[ y1 * npix + x1 ];
                        else
                            image[ y1 * npix + x1 ] += 1.0/fabs(c) * sino[ j * npix + k ];
                    }

                    //  Linear interpolation
                    else{
                        xf  = t/c - y*s/c;
                        w  = xf - floor( xf );
                        x1 = (int)floor( xf ) + nh;
                        y1 = y + nh;

                        if( oper == 0 ){
                            if( x1 >= 0 && x1<= npix-1 )
                                sino[ j * npix + k ] += 1.0/fabs(c) * ( 1 - w ) * image[ y1 * npix + x1 ];
                            if( x1+1 >= 0 && x1+1<= npix-1 ) 
                                sino[ j * npix + k ] += 1.0/fabs(c) * w * image[ y1 * npix + x1 + 1 ];
                        }
                        else{
                            if( x1 >= 0 && x1<= npix-1 ) 
                                image[ y1 * npix + x1 ] += 1.0/fabs(c) * ( 1 - w ) * sino[ j * npix + k ];
                            if( x1+1 >= 0 && x1+1<= npix-1 ) 
                                image[ y1 * npix + x1 + 1 ] += 1.0/fabs(c) * w * sino[ j * npix + k ];
                        }
                    }
                }                                
            }

            
            //  pi/4 < theta < 3pi/4
            else{
                for( x=(int)round(lim[0]) ; x<(int)round(lim[1]) ; x++ ){
                    //  Nearest neighbour interpolation
                    if( method == 0){  
                        y1 = (int)round( t/s - x*c/s ) + nh;
                        x1 = x + nh;
                    
                        if( oper == 0 )
                            sino[ j * npix + k ] += 1.0/fabs(s) * image[ y1 * npix + x1 ];
                        else
                            image[ y1 * npix + x1 ] += 1.0/fabs(s) * sino[ j * npix + k ]; 
                    }

                    //  Linear interpolation
                    else{
                        yf = t/s - x*c/s;
                        w  = yf - floor( yf );
                        y1 = (int)floor( yf ) + nh;
                        x1 = x + nh; 

                        if( oper == 0 ){
                            if( y1 >= 0 && y1<= npix-1 )
                                sino[ j * npix + k ] += 1.0/fabs(s) * ( 1 - w ) * image[ y1 * npix + x1 ];
                            if( y1+1 >= 0 && y1+1<= npix-1 )
                                sino[ j * npix + k ] += 1.0/fabs(s) * w * image[ ( y1 + 1 ) * npix + x1 ];
                        }
                        else{
                            if( y1 >= 0 && y1<= npix-1 )
                                image[ y1 * npix + x1 ] += 1.0/fabs(s) * ( 1 - w ) * sino[ j * npix + k ];
                            if( y1+1 >= 0 && y1+1<= npix-1 )
                                image[ ( y1 + 1 ) * npix + x1 ] += 1.0/fabs(s) * w * sino[ j * npix + k ];
                        }                        
                    }
                }
            }
        }
    }
    free( lim );
}
