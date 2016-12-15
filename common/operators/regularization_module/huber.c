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





float pot_huber_num( float x , float delta ){
    return  x / ( 1.0 + myabs( x / delta ) );
}


float pot_huber_den( float x , float delta ){
    return  1.0 / ( 1.0 + myabs( x / delta ) );
}  



/*
 *   Huber's penalty numerator for
 *   statistical reconstruction
 */

void huber( float *I , int nr , int nc , float delta , char*term , float *R ){
    
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

    if( term[0] == 'n' ){
        for( i=0 ; i < nr ; i++ ){
            for( j=0; j < nc ; j++ ){
                for( k=0 , k1=-1 ; k1 <= 1 ; k1++ ){
                    for( k2=-1 ; k2 <= 1 ; k2++ , k++ ){
                        if( i+k1 >=0 && i+k1 <=nr-1 && j+k2 >=0 && j+k2 <=nc-1){
                            R[ i*nc + j ] += w[k] * pot_huber_num( I[ i*nc + j ] - I[ (i+k1)*nc + j + k2 ] , 
                                                                   delta );
                        }
                    }   
                }
            }
        }
    }
    
    else{
        for( i=0 ; i < nr ; i++ ){
            for( j=0; j < nc ; j++ ){
                for( k=0 , k1=-1 ; k1 <= 1 ; k1++ ){
                    for( k2=-1 ; k2 <= 1 ; k2++ , k++ ){       
                        if( i+k1 >=0 && i+k1 <=nr-1 && j+k2 >=0 && j+k2 <=nc-1){
                            R[ i*nc + j ] += 2 * w[k] * pot_huber_den( I[ i*nc + j ] - I[ (i+k1)*nc + j + k2 ] , 
                                                                       delta );
                        }
                    }   
                }
            }
        }        
    }
    
}
