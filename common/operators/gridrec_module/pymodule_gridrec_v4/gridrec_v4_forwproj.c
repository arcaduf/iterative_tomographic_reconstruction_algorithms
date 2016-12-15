/**********************************************************
 **********************************************************
 ***                                                    ***
 ***             FORWARD REGRIDDING PROJECTOR           ***
 ***                                                    ***
 ***        Written by F. Arcadu on the 19/03/2014      ***
 ***                                                    ***
 **********************************************************
 **********************************************************/




/**********************************************************
 **********************************************************
 ***                                                    ***
 ***                       HEADERS                      ***
 ***                                                    ***
 **********************************************************
 **********************************************************/

#include <math.h>

#ifndef _FILTERS_LIB
#define _FILTERS_LIB
#include "filters.h"
#endif


#include <fftw3.h>



/**********************************************************
 **********************************************************
 ***                                                    ***
 ***                        MACROS                      ***
 ***                                                    ***
 **********************************************************
 **********************************************************/

//#define convolv_nn(X) ( lut[ (int)(X+0.5) ])
//#define convolv_lin(X) ( ( ceil(X) - X ) * lut[ (int) floor(X)] + ( X - floor(X) ) * lut[ (int) ceil(X)] )      
#define myAbs(X) ((X)<0 ? -(X) : (X))
#define C 7.0
#define PI 3.141592653589793




/**********************************************************
 **********************************************************
 ***                                                    ***
 ***           FORWARD REGRIDDING PROJECTOR             *** 
 ***                                                    ***
 **********************************************************
 **********************************************************/

void gridrec_v4_forwproj( float *S , int npix , int nang , float *angles , float *param , float *lut ,
                          float *deapod , float *I , char *fftwfn  ) {
    
  /*
   *   Define variables
   */

  int pdim;             //  Number of operative pixels, either equal to "npix" or to "npix" + zero-padding
  int padleft;          //  Pixel position of the last padding zero on the left of the actual projection
  int padright;         //  Pixel position of the first padding zero on the right of the actual projection  
  int pdim_h;           //  Half number of "pdim" pixels
  int npix1, npix2;     //  Pixels at the beginning and at the end of cproj to be saved in the final sinogram
  int interp;
  int radon_degree;
  int idle1, idle2;

  unsigned long j, k, w, n, index;
  unsigned long iul, ivl, iuh, ivh, iv, iu;
  unsigned long ltbl;   // Number of PSWF elements
  
  long ustart, vstart, ufin, vfin;
  
  float Cdata1R, Cdata1I, Cdata2R, Cdata2I, CtmpR, CtmpI;
  float U, V;
  float lconv;        //  Size of the convolution window
  float lconv_h;       //  Half size of the convolution window
  float rtmp;
  float scaling;      //  Convert frequencies to grid units
  float tblspcg;
  float convolv;
  float ctr;
  float *filter_no;  
  fftwf_complex *cproj;    
  float SIN, COS;
  float *work;
  fftwf_complex *H;
  float x;
  float tmp;
  float norm;
  float oversampl;


  FILE *fp = fopen(fftwfn,"r");
  if(fp){
    fftwf_import_wisdom_from_file(fp); // Load wisdom file for faster FFTW
    fclose(fp);
  }

  
  /*
   *   Get external parameters 
   */
    
  ctr          = (float) param[0];
  idle1        = (int)param[1];
  idle2        = (int)param[2];
  oversampl    = (float) param[3];
  interp       = (int) param[4];
  ltbl         = (int) param[5];
  lconv        = (float) param[6];
  radon_degree = (int) param[7];     



  
  /*
   *   Calculate number of operative padded pixels
   */

  pdim = (int) ( pow( 2 , (int)( ceil( log10( npix )/log10(2) ) ) ) * oversampl );
  while( pdim % 4 != 0 )
      pdim++;
  
  padleft = (int) ( ( pdim - npix ) * 0.5 );
  padright = (int) ( padleft + npix );
  pdim_h = 0.5 * pdim;
  npix1 = (int)( npix * 0.5 ); 
  npix2 = 2 * (int)( pdim - npix1 );




  /*
   *   Allocate memory for complex array storing each
   *   projection and its FFT at time
   */

  cproj = (fftwf_complex *)fftwf_malloc( pdim*sizeof(fftwf_complex));
  for(n=0;n<pdim;n++){
      cproj[n][0]=0;
      cproj[n][1]=0;
  } 


  
  /*
   *   Initialize look-up-table for the PSWF interpolation kernel
   *   and the correction matrix
   */
  
  lconv = (float)( 2*C*1.0 / PI ); 
  lconv_h = lconv * 0.5; 
  tblspcg = 2 * ltbl / lconv;  
  work = (float *)calloc( (int)lconv + 1 , sizeof(float) );   


    
  /*
   *   Allocate memory for cartesian Fourier grid
   */

  H = (fftwf_complex*)fftwf_malloc(pdim*pdim*sizeof(fftwf_complex));
  for(n=0;n<pdim*pdim;n++){
      H[n][0]=0;
      H[n][1]=0;
  }


 
  /*
   *   Correct for center of rotation
   */    

  ctr = npix * 0.5;
  if( pdim != npix )
    ctr += (float) padleft;

  int pdim_d = 2 * pdim;

  filter_no = (float *)calloc( pdim_d , sizeof(float) );
  tmp = (float)( 2 * PI * ctr / pdim );
  norm = (float)( PI / pdim / nang );
  norm = 1.0;

  float tmp1;
  for( j=0,k=0 ; j<pdim ; j+=2,k++ ){
    x = k * tmp;
    float fValue = 1.0;
    filter_no[j] = fValue * norm * cos( x );
    filter_no[j+1] = fValue * norm * sin( x );

    if ( radon_degree ){
        tmp1 = filter_no[j];
        filter_no[j] = filter_no[j+1] * ( 2 * PI * k ) / ( 1.0 * pdim);
        filter_no[j+1] = tmp1 * ( - 2 * PI * k ) / ( 1.0 * pdim );    
    }       
  }



  /*
   *   Multiplication for a correction matrix
   */

  j = 0;
  ustart = pdim - pdim_h;
  ufin = pdim;
  
  while (j < pdim) {
    for( iu = ustart ; iu < ufin ; j++ , iu++ ) {
      //corrn_u = winv[j];
      k = 0;
      vstart = pdim - pdim_h ;
      vfin = pdim;
      
      while( k < pdim )	{
	    for( iv = vstart ; iv < vfin ; k++ , iv++ ) {
	      //corrn = corrn_u * winv[k];

          if( padleft <= j && j < padright && padleft <= k && k < padright ){
	        index = ( npix - 1 - (k-padleft) ) * npix + (j-padleft);
	        H[iu * pdim + iv][0] = deapod[ j * pdim + k ] * I[ index ];

	        //printf("\nj = %d  iu = %d  k = %d  iv = %d  index = %d  deapod = %.9f  I[index] = %.9f",
	        //        j , iu, k , iv, index, deapod[j*pdim+k], I[index]);                
          }
	    }

        if (k < pdim) {
	      vstart = 0; 
	      vfin = pdim_h;
	    }
      }
    }

    if (j < pdim) {
      ustart = 0;
      ufin = pdim_h;
    }
  }



  /*
   *   Perform 2D FFT of the cartesian Fourier Grid
   */

   fftwf_plan p = fftwf_plan_dft_2d(pdim, pdim,
                                    H, H,
                                    FFTW_BACKWARD, FFTW_ESTIMATE);
   fftwf_execute(p);
 


  /*
   *   Interpolation of the cartesian Fourier grid with PSWF
   */
  
  fftwf_plan p1 = fftwf_plan_dft_1d(pdim, cproj, cproj, FFTW_FORWARD, FFTW_ESTIMATE);

  tmp = 1.0/(float)pdim;

  for( n=0 ; n<nang ; n++ ){ 
    
    /*
     *   Loop on half number of harmonics, because hermitianity
     *   is exploited
     */

    for( j=0, w=0 ; j < pdim_h ; j++, w++ ) {  
      
      Cdata1R = 0.0;
      Cdata1I = 0.0;
      Cdata2R = 0.0;
      Cdata2I = 0.0; 
      
      
      /*
       *   Get cartesian neighbouring points for each polar point
       */

      SIN = sin( angles[n] * PI / 180 );
      COS = cos( angles[n] * PI / 180 );
      scaling = 1.0;
      U = ( rtmp = scaling * w ) * COS + pdim_h; 
      V = rtmp * SIN + pdim_h;           
      
      iul = (long)ceil( U - lconv_h ); iuh = (long)floor( U + lconv_h );
      ivl = (long)ceil( V - lconv_h ); ivh = (long)floor( V + lconv_h );
      
      if ( iul<0 ) iul = 0; if ( iuh >= pdim ) iuh = pdim-1;   
      if ( ivl<0 ) ivl = 0; if ( ivh >= pdim ) ivh = pdim-1;

      //printf("\n\nscaling = %.5f  rtmp = %.5f  COS = %.5f  SIN = %.5f  U = %.5f  V = %.5f  iul = %d  iuh = %d  ivl = %d  ivh = %d  tblspcg = %.5f", scaling , rtmp , COS, SIN , U, V , iul , iuh , ivl , ivh , tblspcg);
      

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

      //printf("\nmyAbs( V - iv ) * tblspcg = %.9f" , myAbs( V - iv ) * tblspcg);

      
      /*
       *   Calculate the contribution of all neighbouring cartesian points
       *   to each polar Fourier sample
       */

      for( iu=iul ; iu<=iuh ; iu++ ){
        if( interp == 0)
	        rtmp = convolv_nn( myAbs( U - iu ) * tblspcg , lut );
        else
            rtmp = convolv_lin( myAbs( U - iu ) * tblspcg , lut );
	    
        for( iv=ivl , k=0 ; iv<= ivh ; iv++,k++ ){
	      convolv = rtmp * work[k];
              
          if (iu!=0 && iv!=0 && w!=0){ 
		    Cdata1R += H[iu * pdim + iv][0] * convolv;
            Cdata1I += H[iu * pdim + iv][1] * convolv;
            Cdata2R += H[(pdim-iu) * pdim + pdim - iv][0] * convolv;
            Cdata2I += H[(pdim-iu) * pdim + pdim - iv][1] * convolv;

            //printf("\nj = %d  iu = %d  iv = %d  k = %d  work = %.9f  convolv = %.9f  \
                    H1.r = %.9f  H1.i = %.9f  H2.r = %.9f  H2.i = %.9f  Cdata1R = %.9f  Cdata1I = %.9f  Cdata2R = %.9f  Cdata2I = %.9f",
	        //        j , iu , iv , k , work[k] , convolv , H[iu * pdim + iv][0] , H[iu * pdim + iv][1] ,
            //        H[(pdim-iu) * pdim + pdim - iv][0] , H[(pdim-iu) * pdim + pdim - iv][1],
            //        Cdata1R , Cdata1I, Cdata2R , Cdata2I );                   

          } 
           
          else {
            Cdata1R += H[iu * pdim + iv][0] * convolv;
            Cdata1I += H[iu * pdim + iv][1] * convolv;
            //printf("\nj = %d  iu = %d  iv = %d  k = %d  work = %.9f  convolv = %.9f  \
                    H1.r = %.9f  H1.i = %.9f  Cdata1R = %.9f  Cdata1I = %.9f  Cdata2R = %.9f  Cdata2I = %.9f",
	        //        j , iu , iv , k , work[k] , convolv , H[iu * pdim + iv][0] , H[iu * pdim + iv][1] ,
            //        Cdata1R , Cdata1I, Cdata2R , Cdata2I);  
          }
          //printf("\nwork[%d] = %.9f\n\n", k , work[k]);      
        }
	  }

      
      CtmpR = filter_no[2*j];
      CtmpI = -filter_no[2*j+1];

      if( j!=0 ){
        cproj[pdim-j][0] = CtmpR * Cdata1R  -  CtmpI * Cdata1I;
        cproj[pdim-j][1] = CtmpR * Cdata1I  +  CtmpI * Cdata1R;  

        CtmpI = -CtmpI;
        
        cproj[j][0] = CtmpR * Cdata2R  -  CtmpI * Cdata2I;
        cproj[j][1] = CtmpR * Cdata2I  +  CtmpI * Cdata2R; 
      }

      else {
        cproj[j][0] = CtmpR * Cdata1R  -  CtmpI * Cdata1I;
        cproj[j][1] = CtmpR * Cdata1I  +  CtmpI * Cdata1R;              
      }
    } // End loop on (half) transform data
    
    //for( j=0 ; j<pdim ; j++ )
	//    printf("\nn = %d  j = %d  cproj[%d].r = %.9f  cproj[%d].i = %.9f",
	//            n, j , j, cproj[j][0], j, cproj[j][1]);       

    /*
     *   Perform 1D IFFT of cproj
     */
    
    fftwf_execute(p1); 

    
    /*
     *   Assign sinogram elements
     */
    //  Without adjoint of the zero-padding
    for( k=0 , j=padleft ; k<npix ; j++,k++ )
        S[ n*npix + k ] =  tmp * cproj[j][0];
    
    /*
    printf("\nAngle = %d  Full projection to be assigned\n:", n);
    for( j=0 ; j< pdim ; j++ )
        printf("%.4f  ", tmp * cproj[j][0]); */    


    //  With adjoint of the zero-padding
    /*for( j=0 ; j<padleft+1 ; j++ )
        S[ n*npix ] +=  tmp * cproj[j][0];       

    for( k=1 , j=padleft+1 ; k<npix ; j++,k++ )
        S[ n*npix + k ] =  tmp * cproj[j][0];

    for( ; j< pdim ; j++ )
        S[ n*npix + npix - 1 ] +=  tmp * cproj[j][0];
    */
    /*
    printf("\nAngle = %d  Assigned projection\n:", n);
    for( j=0 ; j< npix ; j++ )
        printf("%.4f  ", S[ n*npix + j ] );*/  

    cproj[pdim_h][0] = 0.0;
    cproj[pdim_h][1] = 0.0;
    
  } // End for loop on angles
 
   fftwf_destroy_plan(p1); 
   fftwf_destroy_plan(p); 



  /*
   *  Free memory
   */

  free( work );
  fftwf_free( cproj );
  fftwf_free( H );
  free( filter_no );

  
  return;
}  
