#include <stdlib.h>
#include <stdio.h>
#include <math.h>



float* penalty_weights(){
    float *w = ( float * ) malloc( 9 * sizeof( float ) );
    
    w[0] = w[2] = w[6] = w[8] = 1.0 / sqrt( 2.0 );
    w[1] = w[3] = w[5] = w[7] = 1.0;
    w[4] = 4.0;
    
    return w;
}
