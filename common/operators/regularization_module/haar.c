/********************************************************************  
 ********************************************************************
 ***                                                              ***
 ***    HUBER PENALTY NUMERATOR FOR STATISTICAL RECONSTRUCTION    ***
 ***                                                              *** 
 ***             Written by F. Arcadu on the 20/02/2015           ***
 ***                                                              ***
 ********************************************************************
 ********************************************************************/



/*
 *   Headers  
 */

#include <stdlib.h>
#include <stdio.h>
#include "penalty_weights.h"




/*
 *   Macros
 */

#define myabs(X) ( (X)<0 ? -(X) : (X) ) 




float pot_l1( float x ){
    if( x >= 0 )
        return x;
    else
        return -x;
}  



/*
 *   Huber's penalty numerator for
 *   statistical reconstruction
 */

void haar( float *I , int nr , int nc , float *R ){
    
    /*
     *   Define variables
     */

    int i, j, k, k1 , k2;



    /*
     *   Pre-compute weights
     */

    float *w = penalty_weights();



    /*
     *   Compute penalty
     */

    for( i=0 ; i < nr ; i++ ){
        for( j=0; j < nc ; j++ ){
            for( k=0 , k1=-1 ; k1 <= 1 ; k1++ ){
                for( k2=-1 ; k2 <= 1 ; k2++ , k++ ){
                    if( i+k1 >=0 && i+k1 <=nr-1 && j+k2 >=0 && j+k2 <=nc-1){
                        R[ i*nc + j ] += w[k] * pot_l1( I[ i*nc + j ] - I[ (i+k1)*nc + j + k2 ] );
                    }
                }   
            }
        }
    }
}
