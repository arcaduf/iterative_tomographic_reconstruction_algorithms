/*
 *  IMPLEMENTATION OF THE PIXEL-DRIVEN TOMOGRAPHIC PROJECTORS 
 */


#include <math.h>
#include <stdio.h>
#include <omp.h>

#define pi 3.141592653589793
#define eps 1e-7



void siddon( float *sino , float *image , float norm , int npix , float *p1 , float *p2 , int index , int oper ){
    int ii, jj, i, j, nh;
    float alpha_old, alpha_x, alpha_y, alpha, dx, dy;
    float incr, l;
    
    nh = (int)( npix * 0.5 );

    incr = sqrt( ( p2[0] - p1[0] ) * ( p2[0] - p1[0] ) + ( p2[1] - p1[1] ) * ( p2[1] - p1[1] ) );

    ii = (int)round( p1[0] );
    jj = (int)round( p1[1] );

    if( p2[0] > p1[0] )
        dx = 1.0;
    else
        dx = -1.0;

    if( p2[1] > p1[1] )
        dy = 1.0;
    else
        dy = -1.0; 


    alpha_old = 0.0;

    while( alpha <= 1 ){ 
        alpha_y = ( jj + 0.5 * dy - p1[1] ) / ( p2[1] - p1[1] );
        alpha_x = ( ii + 0.5 * dx - p1[0] ) / ( p2[0] - p1[0] );
        
        if( alpha_x < alpha_y ){
            alpha = alpha_x;
            ii += dx;
        }
        else if( alpha_x > alpha_y ){
            alpha = alpha_y;
            jj += dy; 
        }
        else if( alpha_x == alpha_y ){
            alpha = alpha_y;
            ii += dx;
            jj += dy;
        }

        l = ( alpha - alpha_old ) * incr;
        i = ii + nh;
        j = jj + nh;

        if( i>0 && i<npix && j>0 && j<npix ){
            if( oper == 0 ) 
                sino[ index ] += 1/5.0 * l * image[ i * npix + j ];
            else
                image[ i * npix + j ] += 1/5.0 * l * sino[ index ];
        }                              

        alpha_old = alpha;
    }
}




void radon_rd( float* image , int npix , float *angles , int nang , int oper , float *sino )
{
    int v, i, j, nh;
    float x1, y1, x2, y2, theta, s, c, t;
    float *p1, *p2;

    nh = ( int )( npix * 0.5 );
    p1 = ( float * )malloc( 2 * sizeof( float ) );
    p2 = ( float * )malloc( 2 * sizeof( float ) ); 

    int index, flag=0;
    float norm = 0.0;


    for( v=0 ; v<nang ; v++ ){
        theta = angles[v];
        s     = sin( theta );
        c     = cos( theta );

        
        if( fabs( s ) < eps ){
            for( i=1 ; i<npix ; i++ ){
                for( j=0 ; j<npix ; j++ ){
                    if( oper == 0 )
                        sino[ v * npix + npix - 1 - i ] += image[ j * npix + i ];
                    else
                        image[ j * npix + i ] += sino[ v * npix + npix - 1 - i ];
                }
            }
        }


        else if( fabs( c ) < eps ){
            for( i=0 ; i<npix ; i++ ){
                for( j=0 ; j<npix ; j++ ){
                    if( oper == 0 )
                        sino[ v * npix + i ] += image[ i * npix + j ];
                    else
                        image[ i * npix + j ] += sino[ v * npix + i ];
                }
            } 
        }


        else{
            index = v * npix;

            for( i=0 ; i< 6 * npix ; i++ ){
                if( i % 6 != 0 ){
                    t = fabs( i/6.0 - nh );

                    if( i >= 3*npix && theta < pi/2 )
                        t = -t;
                    if( i < 3*npix && theta > pi/2 )
                        t = -t;  

                    //  Compute entry and exit point
                    if( theta < pi/2 ){
                        y1 = -nh * s/c + t/c;
                        x1 = -nh * c/s - t/s;
                        y2 = nh * s/c + t/c;
                        x2 = nh * c/s - t/s;

                        if( y1 >= -nh && y1 <= nh ){
                            p1[0] = -nh;
                            p1[1] = y1;
                        }
                        else{
                            p1[0] = x1;
                            p1[1] = -nh;                         
                        }
                        if( y2 >= -nh && y2 <= nh ){
                            p2[0] = nh;
                            p2[1] = y2;
                        }
                        else{
                            p2[0] = x2;
                            p2[1] = nh;                         
                        }                         
                    }

                    else{
                        y1 = nh * s/c - t/c;
                        x1 = -nh * c/s + t/s;
                        y2 = -nh * s/c - t/c;
                        x2 = nh * c/s + t/s;

                        if( y1 >= -nh && y1 <= nh ){
                            p1[0] = nh;
                            p1[1] = y1;
                        }
                        else{
                            p1[0] = x1;
                            p1[1] = -nh;                         
                        }
                        if( y2 >= -nh && y2 <= nh ){
                            p2[0] = -nh;
                            p2[1] = y2;
                        }
                        else{
                            p2[0] = x2;
                            p2[1] = nh;                         
                        }                         
                    }
                    
                    siddon( sino , image , norm , npix , p1 , p2 , index , oper );
                }
                
                else if( i % 6 == 0 && i != 0 )
                    index++;
            }
        }
    }
}
