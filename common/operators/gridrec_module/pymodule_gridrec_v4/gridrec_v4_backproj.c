/********************************************************************  
 ********************************************************************
 ***                                                              ***
 ***         REGRIDDING BACKPROJECTION OPERATOR FOR PYNNGRID      ***
 ***                                                              *** 
 ***             Written by F. Arcadu on the 18/03/2014           ***
 ***                                                              ***
 ********************************************************************
 ********************************************************************/




/********************************************************************
 ********************************************************************
 ***                                                              ***
 ***                             HEADERS                          ***
 ***                                                              ***
 ********************************************************************
 ********************************************************************/

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifndef _FILTERS_LIB
#define _FILTERS_LIB
#include "filters.h"
#endif

#include <fftw3.h>

#include <time.h>




/********************************************************************
 ********************************************************************
 ***                                                              ***
 ***                             MACROS                           ***
 ***                                                              ***
 ********************************************************************
 ********************************************************************/

//#define convolv_nn(X) ( lut[ (int)(X+0.5) ])
//#define convolv_lin(X) ( ( ceil(X) - X ) * lut[ (int) floor(X)] + ( X - floor(X) ) * lut[ (int) ceil(X)] )
#define myAbs(X) ((X)<0 ? -(X) : (X))
#define PI 3.141592653589793
#define C 7.0 




/********************************************************************
 ********************************************************************
 ***                                                              ***
 ***                         BACKPROJECTION                       ***
 ***                                                              ***
 ********************************************************************
 ********************************************************************/

void gridrec_v4_backproj( float *sino , int npix , int nang , float *angles , float *param , 
                          float *lut, float *deapod, float *filter_ext , float *rec, char *fftwfn ) {
    
  /*
   *   Define variables
   */

  int pdim;                 //  Number of zero-padded pixels in order to have a number of pixels power of 2
  int padleft;              //  Pixel position of the last padding zero on the left of the actual projection
  int padright;             //  Pixel position of the first padding zero on the right of the actual projection
  int pdim_d;               //  Double number of "pdim" pixels
  int pdim_h;               //  Half number of "pdim" pixels
  int flag_filter;          //  Flag to specify whether to use the external filter or the standard built-in one
  int type_filter;          //  Specify which standard filter to use
  int interp;
  int time1, time2;
  int radon_degree;

  unsigned long i, j, k, w, n, index;
  unsigned long iul, ivl, iuh, ivh, iv, iu;
  int ltbl;       // Number of PSWF elements
  
  long ustart, vstart, ufin, vfin;
  
  float Cdata1R, Cdata1I, Cdata2R, Cdata2I;
  float Ctmp1R, Ctmp1I, Ctmp2R, Ctmp2I, Ctmp3R, Ctmp3I;     
  float U, V;             // iariables referred to the Fourier cartesian grid
  float lconv;            // Size of the convolution window
  float lconv_h;           // Half size of the convolution window
  float rtmp;
  float scaling;          // Convert frequencies to grid units
  float tblspcg;
  float convolv;
  float ctr;              // Variable to store the center of rotation axis 		  
  fftwf_complex *cproj;           
  float SIN, COS;     
  float *work;
  fftwf_complex *H;
  float *filter_stand;
  float x;
  float tmp;
  float norm;
  float oversampl;

  clock_t time_conv_s, time_conv, time_fft_s, time_fft, time_fft1_s, time_fft1, time_deapod_s, time_deapod;
  time_conv = 0.0;  time_fft = 0.0;  time_fft1 = 0.0;  time_deapod = 0.0;




  /*
   *   Get external parameters 
   */
    
    ctr = (float) param[0];
    type_filter = (int)param[1];
    flag_filter = (int)param[2];
    oversampl = (float) param[3];
    interp = (int) param[4];
    ltbl = (int) param[5];
    lconv = (float) param[6];
    radon_degree = (int) param[7];   




  /*
   *   Open wisdom file for computing FFTW
   */
  
  FILE *fp = fopen(fftwfn,"r");
  if(fp){
    fftwf_import_wisdom_from_file(fp); // Load wisdom file for faster FFTW
    fclose(fp);
  }

  
  /*
   *   Calculate number of operative pixels
   *   This number is equal to smallest power of 2 which >= to the original
   *   number of pixels multiplied by the edge-padding factor 2
   */

  pdim = (int) ( pow( 2 , (int)( ceil( log10( npix )/log10(2) ) )) * oversampl );
  while( pdim % 4 !=0 )
      pdim++;

  padleft = (int) ( ( pdim - npix ) * 0.5 );
  padright = (int) ( padleft + npix );
  pdim_d = 2 * pdim;
  pdim_h = 0.5 * pdim;




  /*
   *   Allocate memory for the complex array storing first each projection
   *   and, then, its Fast Fourier Transform
   */

  time_fft_s = clock();
  cproj = (fftwf_complex *)fftwf_malloc( pdim * sizeof(fftwf_complex));
  for(n=0;n<pdim;n++){
      cproj[n][0]=0;
      cproj[n][1]=0;
  }
     
  time_fft += clock() - time_fft_s;



  /*
   *   Initialize look-up-table for the PSWF interpolation kernel and
   *   for the correction matrix
   */
  
  //lconv = (float)( 2 * C * 1.0 / PI ); 
  lconv_h = lconv * 0.5; 
  tblspcg = 2 * ltbl / lconv;
  work = (float *)calloc( lconv + 1 , sizeof(float) ); 

  
  
  /*
   *   Get center of rotation axis
   */

  if ( ctr == 0 )
    ctr = npix * 0.5;
  if( pdim != npix )
    ctr += (float) padleft;

   

  /*
   *   Use standard filter
   */

  time_fft_s = clock();
  if( flag_filter == 0 ){
    filter_stand = (float *)calloc( pdim_d , sizeof(float) );
    calc_filter( filter_stand , nang , pdim , ctr , type_filter , radon_degree );
  }


  /*
   *   Multiply input filter array per complex exponential in order
   *   to correct for the center of rotation axis
   */
  
  else if( flag_filter == 1){
    tmp = (float)( 2 * PI * ctr / pdim );
    norm = (float)( PI / pdim / nang );
    norm = 1.0;

    for( j=0,k=0 ; j<pdim ; j+=2,k++ ){
        x = k * tmp;
        float fValue = filter_ext[j];
        filter_ext[j] = fValue*norm * cos(x);
        filter_ext[j+1] = -fValue*norm * sin(x);
    }
  }
  time_fft += clock() - time_fft_s;  


  
  /*
   *   Allocate memory for cartesian Fourier grid
   */
  
  time_fft_s = clock();
  H = (fftwf_complex*)fftwf_malloc(pdim*pdim*sizeof(fftwf_complex));
  for(n=0;n<pdim*pdim;n++){
      H[n][0]=0;
      H[n][1]=0;
  }
  
  fftwf_plan p1 = fftwf_plan_dft_1d(pdim, cproj, cproj, FFTW_FORWARD, FFTW_ESTIMATE);
  fftwf_plan p =  fftwf_plan_dft_2d(pdim, pdim, H, H,
                                    FFTW_BACKWARD, FFTW_ESTIMATE); 
  time_fft += clock() - time_fft_s;

  
  
  /*
   *   Interpolation of the polar Fourier grid with PSWF 
   */

  time_conv_s = clock();
  float tmp2 = 1.0/pdim;
  
  for( n=0 ; n<nang ; n++ ){ 

    /*
     *   Store each projection inside "cproj" and, at the same
     *   time, perform the edge-padding of the projection
     */
    
    i = 0;
    j = 0;
    k = 0;

    while( i < padleft ){
      if ( radon_degree )
        cproj[j][0] = 0.0 ;
      else
        cproj[j][0] = sino[ n * npix ];
      cproj[j][1] = 0.0;
      i += 1;
      j += 1;
    }

    while( i < padright ){
      cproj[j][0] = sino[ n * npix + k ];   
      cproj[j][1] = 0.0;
      i += 1;
      j += 1;
      k += 1;
    }

    while( i < pdim ){
      if ( radon_degree )
        cproj[j][0] = 0.0 ;
      else
        cproj[j][0] = sino[ n * npix + npix - 1 ];  
      cproj[j][1] = 0.0;
      i += 1;
      j += 1;
    }



    /*
     *   Perform 1D FFT of the projection
     */
     
     time_fft1_s = clock();
     fftwf_execute(p1);
     time_fft1 += clock() - time_fft1_s;

    
    /*
     *   Loop on the first half Fourier components of
     *   each FFT-transformed projection (one exploits
     *   here the hermitianity of the FFT-array related 
     *   to an original real array)
     */

    for( j=0, w=0 ; j < pdim_h ; j++, w++ ) {  
      
      if( flag_filter != 2 ){
	    Ctmp1R = filter_stand[2*j];
	    Ctmp1I = filter_stand[2*j+1];
      }
      else if( flag_filter == 2 ){
	    Ctmp1R = filter_ext[2*j];
	    Ctmp1I = filter_ext[2*j+1];       
      }
	
	  Ctmp2R = tmp2 * cproj[j][0];
	  Ctmp2I = tmp2 * cproj[j][1];
	
	  if( j!=0 ){
	    Ctmp3R = tmp2 * cproj[pdim-j][0];
	    Ctmp3I = tmp2 * cproj[pdim-j][1];
	
        Cdata1R = Ctmp1R * Ctmp3R  -  Ctmp1I * Ctmp3I;
        Cdata1I = Ctmp1R * Ctmp3I  +  Ctmp1I * Ctmp3R;

        Ctmp1I = -Ctmp1I;

        Cdata2R = Ctmp1R * Ctmp2R  -  Ctmp1I * Ctmp2I;
        Cdata2I = Ctmp1R * Ctmp2I  +  Ctmp1I * Ctmp2R;
	  }

	  else {
	    Cdata1R = Ctmp1R * Ctmp2R;
	    Cdata1I = Ctmp1R * Ctmp2I;  
	    Cdata2R = 0.0;
	    Cdata2I = 0.0;
	  }    


      /*
       *   Get polar coordinates
       */
      SIN = sin( angles[n] * PI / 180 );
      COS = cos( angles[n] * PI / 180 );
      scaling = 1.0;
      U = ( rtmp = scaling * w ) * COS + pdim_h; 
      V = rtmp * SIN + pdim_h;


      /*
       *   Get interval of the cartesian coordinates
       *   of the points receiving the contribution from
       *   the selected polar Fourier sample
       */
     
      iul = (long)ceil( U - lconv_h ); iuh = (long)floor( U + lconv_h );
      ivl = (long)ceil( V - lconv_h ); ivh = (long)floor( V + lconv_h );

      if ( iul<0 ) iul = 0; if ( iuh >= pdim ) iuh = pdim-1;   
      if ( ivl<0 ) ivl = 0; if ( ivh >= pdim ) ivh = pdim-1;
     

      /*
       *   Get convolvent values with nearest neighbour
       *   or linear interpolation
       */

      if( interp == 0 ){
        for (iv = ivl, k=0; iv <= ivh; iv++, k++)
	        work[k] = convolv_nn( myAbs( V - iv ) * tblspcg , lut );
      }
      else{
        for (iv = ivl, k=0; iv <= ivh; iv++, k++)
	        work[k] = convolv_lin( myAbs( V - iv ) * tblspcg , lut );         
      }

     /*
      *   Calculate the contribution of each polar Fourier point
      *   for all the neighbouring cartesian Fourier points 
      */
    
      for( iu=iul ; iu<=iuh ; iu++ ){
        if( interp == 0)
	        rtmp = convolv_nn( myAbs( U - iu ) * tblspcg , lut );
        else
            rtmp = convolv_lin( myAbs( U - iu ) * tblspcg , lut );
	    
        for( iv=ivl , k=0 ; iv<= ivh ; iv++,k++ ){
	      convolv = rtmp * work[k];
          
          if (iu!=0 && iv!=0 && w!=0) { 
		    H[iu * pdim + iv][0] += convolv * Cdata1R;
            H[iu * pdim + iv][1] += convolv * Cdata1I;
            H[(pdim-iu) * pdim + pdim - iv][0] += convolv * Cdata2R;
            H[(pdim-iu) * pdim + pdim - iv][1] += convolv * Cdata2I;
          } 
          else {
            H[iu * pdim + iv][0] += convolv * Cdata1R;
            H[iu * pdim + iv][1] += convolv * Cdata1I;
          }
        } // End loop on y-coordinates of the cartesian neighbours 
	  } // End loop on x-coordinates of the cartesian neighbours
    } // End loop on transform data   
  } // End for loop on angles

  fftwf_destroy_plan(p1); 

  time_conv += clock() - time_conv_s - time_fft1;
  time_fft += time_fft1;



  /*
   *   Perform 2D FFT of the cartesian Fourier Grid
   */
  
   time_fft_s = clock();
   //fftwf_plan p =  fftwf_plan_dft_2d(pdim, pdim, H, H,
   //                                  FFTW_BACKWARD, FFTW_ESTIMATE);
   fftwf_execute(p);
   time_fft += clock() - time_fft_s;

   fftwf_destroy_plan(p); 



  /*
   *  Multiplication for the PSWF correction matrix
   */

  
  time_deapod_s = clock();
  j = 0;
  ustart = pdim - pdim_h;
  ufin = pdim;
 
  while (j < pdim){    
    for( iu = ustart ; iu < ufin ; j++ , iu++ ){
      //corrn_u = winv[j];
      k = 0;
      vstart = pdim - pdim_h ;
      vfin = pdim;
      
      while( k < pdim ){
	    for( iv = vstart ; iv < vfin ; k++ , iv++ ) {
	      //corrn = corrn_u * winv[k];
 
	      /*
           *   Select the centered square npix * npix
           */
 
     	  if( padleft <= j && j < padright && padleft <= k && k < padright ){
	        index = ( npix - 1 - ( k-padleft) ) * npix + (j-padleft);
	        rec[index] = deapod[j * pdim + k] * H[iu * pdim + iv][0];

            /*
            if ( index == ( (int)( npix * 0.5 ) - 1 ) * npix + ( (int)( npix * 0.5 ) -1 ) ){
                printf("\nImage apod ind. %d :  %.9f", H[iu * pdim + iv][0], index);
                printf("\nImage deapod ind. %d : %.9f", rec[index], index);  
            }
            if ( index == ( 1430 - 1 ) * npix + ( 1027 - 1 ) ){
                printf("\nImage apod ind. %d :  %.9f", H[iu * pdim + iv][0], index);
                printf("\nImage deapod ind. %d : %.9f", rec[index], index);  
            }
            if ( index == ( 1944 - 1 ) * npix + ( 1024 - 1 ) ){
                printf("\nImage apod ind. %d :  %.9f", H[iu * pdim + iv][0], index);
                printf("\nImage deapod ind. %d : %.9f", rec[index], index);  
            }
            */
	      }	  
	    }
      
	    if (k < pdim) {
	      vstart = 0; 
      	  vfin = pdim_h;
      	}
      } // End while loop on k
    } // End for loop on iu

    if (j < pdim) {
      ustart = 0;
      ufin = pdim_h;
    }
  } // End while loop on j

  time_deapod = clock() - time_deapod_s;




  /*
   *  Free memory
   */
  
  fftwf_free( cproj );
  fftwf_free( H );
  free( work );
  if( flag_filter == 0 )
      free( filter_stand );



  return;
}




void create_fftw_wisdom_file(char *fn, int npix){
    int padfactor = 2;
    int pdim = (int) pow( 2 , (int)( ceil( log10( npix )/log10(2) ) )) * padfactor;
    fftwf_complex *H = (fftwf_complex*)fftwf_malloc(pdim*pdim*sizeof(fftwf_complex));
    fftwf_complex *H2 = (fftwf_complex*)fftwf_malloc(pdim*sizeof(fftwf_complex));
    FILE *fp = fopen(fn,"r");
    if(fp){
        fftwf_import_wisdom_from_file(fp); // Load wisdom file for faster FFTW
        fclose(fp);
    }
    fftwf_plan p =  fftwf_plan_dft_2d(pdim, pdim,
                                H, H,
                                FFTW_BACKWARD, FFTW_MEASURE);
    fftwf_plan p1 =  fftwf_plan_dft_1d(pdim,
                                H2,H2,
                                FFTW_FORWARD, FFTW_MEASURE);
                                
    fp = fopen(fn,"w");
    if(fp){
        fftwf_export_wisdom_to_file(fp);
        fclose(fp);
    }
    fftwf_destroy_plan(p);
    fftwf_destroy_plan(p1);
    fftwf_free(H);
    fftwf_free(H2);
}
