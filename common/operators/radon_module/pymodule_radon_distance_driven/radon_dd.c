/*
 *  IMPLEMENTATION OF THE PIXEL-DRIVEN TOMOGRAPHIC PROJECTORS 
 */




#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define pi 3.141592653589793
#define eps 1.0e-7


typedef struct{
        float xcoor;
        short int label;
        short int index;      
} Proj;




void quick_sort( Proj *a , int n ) {
    int i, j;
    float p;
    Proj t;

    if ( n < 2 )
        return;

    p = a[n/2].xcoor;

    for( i = 0 , j=n-1;; i++ , j-- ){
        while( a[i].xcoor < p )
            i++;
        while( p < a[j].xcoor )
            j--;
        if( i >= j )
            break;
        t = a[i];
        a[i] = a[j];
        a[j] = t;
    }
    quick_sort( a , i );
    quick_sort( a + i , n - i );
}




void radon_dd( float* image , int npix , float *angles , int nang , int oper , float *sino )
{
    int v, i, j, k, nh, nt, flag1, flag2, i_d, i_p;
    float s, c, x, y, theta, xi, xd, diff;
    float *xcoor;
    int *label, *index;

    
    nh = ( int )( npix * 0.5 );
    nt = 2 * ( npix + 1 );

    
    Proj *proj = ( Proj * )malloc( nt * sizeof( Proj ) ); 

    
    for( v=0 ; v<nang ; v++ ){
        theta = angles[v];
        s     = sin( theta );
        c     = cos( theta );

        
        if( fabs( s ) < eps  ){
            for( i=0 ; i<npix ; i++ ){
                for( j=0 ; j<npix ; j++ ){
                    if( oper == 0 )
                        sino[ ( nang - 1 - v ) * npix + i ] += image[ i * npix + j ];
                    else
                        image[ i * npix + j ] += sino[ ( nang - 1 - v ) * npix + i ]; 
                }
            }    
        }
        
        
        else if( fabs( c ) < eps ){
            for( i=0 ; i<npix ; i++ ){
                for( j=0 ; j<npix ; j++ ){
                    if( oper == 0 )
                        sino[ ( nang - 1 - v ) * npix + i ] += image[ j * npix + i ];
                    else
                        image[ j * npix + i ] += sino[ ( nang - 1 - v ) * npix + i ]; 
                }
            }    
        }              
         
        
        else if( theta > pi/2 ){ 
            for( i=0 ; i<npix ; i++ ){
                y  = i - nh + 0.5;

                for( j=0 ; j<npix+1 ; j++ ){
                    x  = j - nh;

                    proj[2*j].xcoor   = x - y * c / s;
                    proj[2*j].label   = 0;
                    proj[2*j].index   = j;
                    if( theta < pi/2 )
                        proj[2*j+1].xcoor = - x / s;
                    else
                        proj[2*j+1].xcoor = x / s; 
                    proj[2*j+1].label = 1;
                    proj[2*j+1].index = j;
                }

                quick_sort( proj , nt );
                
                flag1 = 0;
                flag2 = 0;
                i_p   = -1;
                i_d   = -1;
                diff  = -1;

                for( j=1 ; j<nt ; j++ ){
                    if( proj[j].label == 1 && proj[j-1].label == 0 ){
                        if( flag1 == 0 )
                            flag1 = 1;
                        else{
                            i_d = proj[j].index - 1;
                            i_p = proj[j-1].index;
                            diff = fabs( proj[j].xcoor - proj[j-1].xcoor );
                            flag2 = 1;
                        }
                    }

                    else if( proj[j].label == 0 && proj[j-1].label == 1 ){
                        if( flag1 == 0 )
                            flag1 = 1;
                        else{                         
                            i_p = proj[j].index - 1;
                            i_d = proj[j-1].index;
                            diff = fabs( proj[j].xcoor - proj[j-1].xcoor );
                            flag2 = 1;
                        }
                    }

                    else if( proj[j].label == 0 && proj[j-1].label == 0 ){
                        for( k=j-1 ; k>=0 ; k-- ){
                            if( proj[k].label == 1 && proj[k].index != npix ){
                                flag2 = 1;
                                break;
                            }
                        }
                        if( flag2 ){                        
                            i_p = proj[j-1].index;
                            i_d = proj[k].index;
                            diff = fabs( proj[j].xcoor - proj[j-1].xcoor );                         
                        }
                    }

                    else if( proj[j].label == 1 && proj[j-1].label == 1 ){
                        for( k=j-1 ; k>=0 ; k-- ){
                            if( proj[k].label == 0 && proj[k].index != npix ){
                                flag2 = 1;
                                break;
                            }
                        }
                        if( flag2 ){                        
                            i_d = proj[j-1].index;
                            i_p = proj[k].index;
                            diff = fabs( proj[j].xcoor - proj[j-1].xcoor );                         
                        }
                    }

                    if( proj[j].index == npix )
                        break;                      

                    if( flag2 ){ 
                        if( oper == 0 )
                            sino[ ( nang -1 - v ) * npix + i_d ] += diff * image[ i * npix + i_p ];
                        else
                            image[ i * npix + i_p ] += diff * sino[ ( nang -1 - v ) * npix + i_d ];
                        flag2 = 0;
                        i_d   = -1;
                        i_p   = -1;
                        diff  = -1;
                    }
                }
            }
        } 
        
        
        else if( theta < pi/2.0 && theta != 0.0 ){ 
            for( i=0 ; i<npix ; i++ ){
                x  = i - nh + 0.5;

                for( j=0 ; j<npix+1 ; j++ ){
                    y  = j - nh;

                    proj[2*j].xcoor   = y - x * s / c;
                    proj[2*j].label   = 0;
                    proj[2*j].index   = j;
                    proj[2*j+1].xcoor = y / c;
                    proj[2*j+1].label = 1;
                    proj[2*j+1].index = j;
                }

                quick_sort( proj , nt );
                
                flag1 = 0;
                flag2 = 0;
                i_p   = -1;
                i_d   = -1;
                diff  = -1;

                for( j=1 ; j<nt ; j++ ){
                    if( proj[j].label == 1 && proj[j-1].label == 0 ){
                        if( flag1 == 0 )
                            flag1 = 1;
                        else{
                            i_d = proj[j].index - 1;
                            i_p = proj[j-1].index;
                            diff = fabs( proj[j].xcoor - proj[j-1].xcoor );
                            flag2 = 1;
                        }
                    }

                    else if( proj[j].label == 0 && proj[j-1].label == 1 ){
                        if( flag1 == 0 )
                            flag1 = 1;
                        else{                         
                            i_p = proj[j].index - 1;
                            i_d = proj[j-1].index;
                            diff = fabs( proj[j].xcoor - proj[j-1].xcoor );
                            flag2 = 1;
                        }
                    }

                    else if( proj[j].label == 0 && proj[j-1].label == 0 ){
                        for( k=j-1 ; k>=0 ; k-- ){
                            if( proj[k].label == 1 && proj[k].index != npix ){
                                flag2 = 1;
                                break;
                            }
                        }
                        if( flag2 ){                        
                            i_p = proj[j-1].index;
                            i_d = proj[k].index;
                            diff = fabs( proj[j].xcoor - proj[j-1].xcoor );                         
                        }
                    }

                    else if( proj[j].label == 1 && proj[j-1].label == 1 ){
                        for( k=j-1 ; k>=0 ; k-- ){
                            if( proj[k].label == 0 && proj[k].index != npix ){
                                flag2 = 1;
                                break;
                            }
                        }
                        if( flag2 ){                        
                            i_d = proj[j-1].index;
                            i_p = proj[k].index;
                            diff = fabs( proj[j].xcoor - proj[j-1].xcoor );                         
                        }
                    }

                    if( proj[j].index == npix )
                        break;                      

                    if( flag2 ){ 
                        if( oper == 0 )
                            sino[ ( nang -1 - v ) * npix + i_d ] += diff * image[ i_p * npix + i ];
                        else
                            image[ i_p * npix + i ] += diff * sino[ ( nang -1 - v ) * npix + i_d ];
                        flag2 = 0;
                        i_d   = -1;
                        i_p   = -1;
                        diff  = -1;
                    }
                }
            }          
        }
    }
}
