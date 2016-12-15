//#ifndef _FFT_C
//#define _FFT_C  
//#include "fft.h"
//#endif

#include <fftw3.h>

#define PI 3.141592653589793   
#define myAbs(X) ((X)<0 ? -(X) : (X))



/*
 * RAMP FILTER
 */

void ramp( int size , fftwf_complex *ramp_array ) {
	
    int i, c;
    int sizeH = (int)( size * 0.5 );
  
    for ( i=0 ; i<size ; i++ ) {
        c = i - ( sizeH-1 );
    
    /* Real component */
    if ( c == 0 ) 
      ramp_array[i][0] = 0.25;
    
    else if ( c % 2 == 0)
      ramp_array[i][0] = 0.0;
    
    else
      ramp_array[i][0] = -1 / ( PI*PI*c*c );
    
    /* Imaginary component */
    ramp_array[i][1] = 0;
  }
}



/*
 *  SHEPP-LOGAN FILTER
 */

float shepp_logan( float x , float fm ){
    float ffm;
  
    ffm = x/(2.*fm);
  
    if ( x==0 )
        return 1.0;
  
    else if ( x<fm )
        return (float)( sin( PI * ffm )/( PI * ffm ) );
  
    else
        return 0.0;
}



/*
 *  HANNING FILTER
 */
     
float hanning( float x , float fm ){
    float ffm;
	ffm = x/fm;

    if ( myAbs(x) <= fm )
	    return (float)( 0.5 * ( 1.0 + cos( PI*ffm ) ) ); 

    else
	    return 0.0;
}



/*
 *  HAMMING FILTER
 */  

float hamming( float x , float fm ){
    float ffm;
	ffm = x/fm;
	
    if ( myAbs(x) <= fm )
	  return (float)( 0.54 + 0.46 * cos( PI*ffm ) ); 
    
    else
	  return 0.0;
} 



/*
 * LANCZOS
 */

float lanczos( float x , float fm ){
    float ffm;
	ffm = x/fm;

    if ( x == 0 )
	    return 1.0;
    
    else if ( x < fm )
	    return (float)( sin( PI*ffm ) / ( PI*ffm ) ); 

    else
	    return 0.0;
}



/*
 * PARZEN FILTER
 */

float parzen( float x , float fm ){
    float ffm;
  
    ffm = myAbs( x )/fm;
  
    if ( myAbs( x ) <= fm/2.)  
        return (float)( 1 - 6*ffm*ffm*( 1 - ffm ));
  
    else if ( myAbs( x ) > fm/2. && myAbs( x ) <= fm ) 
        return (float)( 2 * (1-ffm) * (1-ffm) * (1-ffm) );
  
    else 
        return 0.0;
}




/*
 * FILTER FUNCTION
 */

void calc_filter( float *filter, unsigned long nang, unsigned long N , float center , int type_filter , 
                  int radon_degree )
{ 
    long i; long j; long k; 
    long N2 =  (long)(N * 0.5);    
    float x; 
    float filter_weight = 1.0;
    float rtmp1 = (float)( 2*PI*center/N ); 
    float rtmp2;
    float norm = (float)( PI/N/nang );
    fftwf_complex *ramp_array; 
    float fm = 0.5;
    float tmp1;

  
 
    /*
     *   Create ramp filter if some filtering is selected
     */
        
    if( type_filter && radon_degree==0 ){
        ramp_array = (fftwf_complex *)fftwf_malloc( N * sizeof(fftwf_complex) ); 
        
        for( i=0 ; i<N ; i++ ){
	        ramp_array[i][0] = 0.0;
            ramp_array[i][1] = 0.0; 
        }
      
        ramp( N , ramp_array );

        fftwf_plan p1 = fftwf_plan_dft_1d( N , ramp_array , ramp_array ,
                                           FFTW_FORWARD , FFTW_ESTIMATE );
        fftwf_execute(p1);
        fftwf_destroy_plan(p1);

        for ( i=0 ; i<N2 ; i++ )
	        ramp_array[i][0] = sqrt( ( ramp_array[i][0] * ramp_array[i][0] ) +	\
		                        ( ramp_array[i][1] * ramp_array[i][1] ) );

        while( i<N ){
	        ramp_array[i][0] = 0.0;
	        ramp_array[i][1] = 0.0; 
	        i++;
        }
    }


  
    /*
    * Choose filter to superimpose to the ramp one
    */

    float ( *filter_add )( float , float );

    // Shepp-Logan filter_add
    if ( type_filter == 2 )
        filter_add = shepp_logan;

    // Hanning filter_add
    else if ( type_filter == 3 )
        filter_add = hanning;

    // Hamming filter_add
    else if ( type_filter == 4 )
        filter_add = hamming;

    // Lanczos filter_add
    else if ( type_filter == 5 )
        filter_add = lanczos;

    // Parzen filter_add
    else if ( type_filter == 6 )
        filter_add = parzen;



    /*
     *  Create reconstruction filter_add + center of rotation corr.
     */
  
  for( j=0,k=0 ; j<N ; j+=2,k++ ){
    x = k * rtmp1;

    if ( radon_degree == 0 ){
        if ( type_filter && type_filter != 1 )
            rtmp2 = ( 1 - filter_weight + filter_weight * ( *filter_add )( (float)k/N , fm ) \
                    * ramp_array[k][0] ) * norm * N;
        else if ( type_filter == 1 )
            rtmp2 = ( 1 - filter_weight + filter_weight * ramp_array[k][0] ) * norm * N;
        else
            rtmp2 = 1.0;
    }
    else{
        if ( type_filter && type_filter != 1 )
            rtmp2 = ( 1 - filter_weight + filter_weight * ( *filter_add )( (float)k/N , fm ) ) * norm * N;
        else if ( type_filter == 1 )
            rtmp2 = norm * N;
        else
            rtmp2 = 1.0;   
    }

    filter[j] = rtmp2 * cos(x);
    filter[j+1] = -rtmp2 * sin(x);
            
    if( radon_degree ){
        if ( type_filter ){
            rtmp2 = ( 1 - filter_weight + filter_weight ) * norm * N;
            if ( j>0 ){
                tmp1 =  filter[j];
                filter[j] = filter[j+1] * ( -1 ) / ( 2.f * PI );
                filter[j+1] = tmp1 / ( 2.f * PI );
            }
            else{
                filter[j] = 0.0;
                filter[j+1] = 0.0;
            }
        }
        else{
            if ( j>0 ){
                tmp1 = filter[j];
                filter[j] = filter[j+1] * ( - 2 * PI * k ) / ( 1.0 * N );
                filter[j+1] = tmp1 * ( 2 * PI * k ) / ( 1.0 * N );
            }
            else{
                filter[j] = 0.0;
                filter[j+1] = 0.0;               
            }
        }
    }
  }       


  if ( type_filter && radon_degree == 0 )
    fftwf_free( ramp_array );
}




/*
 *   KERNEL INTERPOLATION METHODS
 */

 float convolv_nn( float x , float *lut ){
    return lut[ (int) round( x ) ];
 }


 float convolv_lin( float x , float *lut ){
    float x_inf, d;
    x_inf = floor( x );
    d = x - x_inf;
    return ( 1 - d ) * lut[ (int)x_inf ] + d * lut[ (int)x_inf + 1 ];
}
