
/***************************************************************************/
/* Name:          SBNLTV_mex.c                                             */
/* Description:                                                            */
/* Date:          09-02-10                                                 */
/* Author:        Xavier Bresson (xbresson@math.ucla.edu)                  */
/***************************************************************************/


/* mex SBNLTV_mex.c */


#include <stdio.h>
#include <stdlib.h>
//#include <mex.h>
#include <math.h>
#include <time.h>


 
#define YES 0
#define NO 1

#define X(ix,iy) (ix)*iNy+ (iy)
#define Xinv(ix,iy) (iy)*iNx+ (ix)
#define X4(iXNeigh4,i) iXNeigh4 + i

#define SQR(x) (x)*(x)



float sqrtf(float number) {
    long i;
    float x, y;
    const float f = 1.5F;
    
    x = number * 0.5F;
    y  = number;
    i  = * ( long * ) &y;
    i  = 0x5f3759df - ( i >> 1 );
    y  = * ( float * ) &i;
    y  = y * ( f - ( x * y * y ) );
    y  = y * ( f - ( x * y * y ) );
    return number * y;
}











/****************************************/
//extern void mexFunction(int iNbOut, mxArray *pmxOut[],
//int iNbIn, const mxArray *pmxIn[])
void sbnltv( float *pfu , float *pfIm0 , float *pfW, int *piY, int *piSY, float *param , float *pfuNew )  
{
    
  /* iNbOut: number of outputs
     pmxOut: array of pointers to output arguments */
    
  /* iNbIn: number of inputs
     pmxIn: array of pointers to input arguments */
    
    
    float   fLambda, fMu, fw;
    float   fdij, fdji;
    float   *pfdNew, *pfd, fctST, fSum1, fSum2, fDivWD, fDen, fNormU, fTemp;
    float   *pfbNew, *pfuTemp, fSqrtw, fGuij;
    float   *pfTemp, *pfb, fbij, fbji, *pfGu;
    int     iNy, iNx, iNdim, iDim[3], iDisplay;
    int     iNbNeigh, im, im2, iw, iw2, ic1;
    int     iy, ix, i;
    int     iNbIters, iIter, ixY, iyY;
    int     iNyx, iX, iXb, iIterU, iNbInnerIter, iIncludeCloseNeigh;
    int     iNbNeigh4, iNbNeigh2, iXNeigh4, iYNeigh4, iXNeigh2, iYNeigh2, iY, i2;
    int     iNbNeighX;
    time_t  start_time, end_time;
    
    
    start_time = clock();
    
    
    //printf("\nStart NL-TV with Split-Bregman\n");
    
    
    
    //pfu = mxGetData(pmxIn[0]);
    
    //pfd = mxGetData(pmxIn[1]);
    
    //pfb = mxGetData(pmxIn[2]);
    
    //pfIm0 = mxGetData(pmxIn[3]);
    
    //pfW = mxGetData(pmxIn[4]);
    
    //piY = mxGetData(pmxIn[5]);
    
    //piSY = mxGetData(pmxIn[6]);
    
    //param = mxGetData(pmxIn[7]);
    
    
  /* Get the displaying's indicator of different messages, values, etc */
    iNx = (int) param[0];
    iNy = (int) param[1];
    im = (int) param[2];
    iw = (int) param[3];
    iNbNeigh = (int) param[4];
    fLambda = param[5];
    fMu = param[6];
    iNbIters = (int) param[7];
    iNbInnerIter = (int) param[8];
    iIncludeCloseNeigh = (int) param[9];
    //printf("iNy= %i, iNx= %i, im= %i, iw= %i, iNbNeigh= %i\n",iNy,iNx,im,iw,iNbNeigh);
    //printf("fLambda= %.6f, fMu= %.6f, iNbIters= %i, iNbInnerIter= %i\n",fLambda,fMu,iNbIters,iNbInnerIter);
    
    pfu = (float *) calloc( iNx * iNy * iNbNeigh , sizeof( float ) );
    pfb = (float *) calloc( iNx * iNy * iNbNeigh , sizeof( float ) ); 
    
    if (iNbNeigh>iw*iw-4) 
    {
        iNbNeigh = iw*iw;
        iIncludeCloseNeigh = NO;
    }
    else
        if ( iIncludeCloseNeigh==YES ) iNbNeigh += 4;
    //printf("iNbNeigh= %i, iIncludeCloseNeigh= %i\n",iNbNeigh,iIncludeCloseNeigh);

    
    
    im2 = (im-1)/2;
    iw2 = (iw-1)/2;
    ic1 = im2+iw2;
    
    iNyx = iNy* iNx;
    
    fctST = 1./ fLambda;
    
    iNbNeigh4 = 4*iNbNeigh;
    iNbNeigh2 = 2*iNbNeigh;
    
    
    
    
    
    
    pfd    = ( float *) calloc( iNx * iNy * 2 * iNbNeigh , sizeof( float ) );
    pfb    = ( float *) calloc( iNx * iNy * 2 * iNbNeigh , sizeof( float ) );    
    pfdNew = ( float *) calloc( iNx * iNy * 2 * iNbNeigh , sizeof( float ) );
    pfbNew = ( float *) calloc( iNx * iNy * 2 * iNbNeigh , sizeof( float ) ); 
    pfTemp = ( float *) calloc( iNx * iNy * 10 , sizeof( float ) );     
    
    
    //iNdim = 2;
    //iDim[0] = iNy;
    //iDim[1] = iNx;
    
    //pmxOut[0] = mxCreateNumericArray(iNdim,(const int*)iDim,mxSINGLE_CLASS,mxREAL);
    //pfuNew = mxGetData(pmxOut[0]);
    
    
    //iNdim = 3;
    //iDim[0] = iNx;
    //iDim[1] = iNy;
    //iDim[2] = 2*iNbNeigh;
    
    //pmxOut[1] = mxCreateNumericArray(iNdim,(const int*)iDim,mxSINGLE_CLASS,mxREAL);
    //pfdNew = mxGetData(pmxOut[1]);
    
    //pmxOut[2] = mxCreateNumericArray(iNdim,(const int*)iDim,mxSINGLE_CLASS,mxREAL);
    //pfbNew = mxGetData(pmxOut[2]);
    
    //iNdim = 3;
    //iDim[0] = iNy;
    //iDim[1] = iNx;
    //iDim[2] = 10;
    
    //pmxOut[3] = mxCreateNumericArray(iNdim,(const int*)iDim,mxSINGLE_CLASS,mxREAL);
    //pfTemp = mxGetData(pmxOut[3]);
    
    
    
    
   
    
    
    
    
    pfuTemp = (float *) calloc( (unsigned)(iNy*iNx), sizeof(float) );
    if (!pfuTemp)
        printf("Memory allocation failure\n");
    
    pfGu = (float *) calloc( (unsigned)(2*iNbNeigh), sizeof(float) );
    if (!pfGu)
        printf("Memory allocation failure\n");
    
    
    
    
    // init
    for (i=0; i< iNyx; i++)
    {
        pfuNew[i] = pfu[i];
        pfuTemp[i] = pfu[i];
        for (i2=0; i2< iNbNeigh2; i2++) 
        { pfdNew[i2*iNyx+i] = pfd[i2*iNyx+i]; pfbNew[i2*iNyx+i] = pfb[i2*iNyx+i]; }
    }

    
    
    
    // iterations
    for (iIter=0; iIter< iNbIters; iIter++)
    {
        
        // u
        for (iIterU=0; iIterU< iNbInnerIter-1; iIterU++)
        {
            for (iy=0; iy< iNy; iy++)
                for(ix=0; ix< iNx; ix++)
            {
                iX = X(ix,iy);
                iXb = Xinv(ix,iy);
                iXNeigh4 = iX*iNbNeigh4;
                iXNeigh2 = iX*iNbNeigh2;
                fSum1 = 0.0;
                fSum2 = 0.0;
                fDivWD = 0.0;
                iNbNeighX = piSY[iXb];
                for (i=0; i<iNbNeighX; i++)
                {
                    ixY = piY[X4(iXNeigh4,2*i)];
                    iyY = piY[X4(iXNeigh4,2*i+1)];
                    iY = X(ixY,iyY);
                    iYNeigh4 = iY* iNbNeigh4;
                    iYNeigh2 = iY* iNbNeigh2;
                    
                    fw = pfW[X4(iXNeigh4,2*i)];
                    fSqrtw = pfW[X4(iXNeigh4,2*i+1)];
                    
                    fdij = pfdNew[X4(iXNeigh2,i)];
                    fbij = pfbNew[X4(iXNeigh2,i)];
                    fdji = pfdNew[X4(iYNeigh2,i)];
                    fbji = pfbNew[X4(iYNeigh2,i)];
                    fSum1 += fw;
                    fSum2 += fw* pfuNew[iY];
                    fDivWD += fSqrtw*( fdij-fdji );
                    fDivWD -= fSqrtw*( fbij-fbji );
                    
                } // end for (i=0; i<piSY[iXb]; i++)
                fDen = fMu + 2.0*fLambda*fSum1; // 2 in graph Laplacian
                pfuNew[iX] = (2.0*fLambda*fSum2 + fMu*pfIm0[iX] - fLambda*fDivWD)/ fDen;
                
                } // end for iy,ix 
            
        } // end for (iIterU=0; iIterU< iNbInnerIter-1; iIterU++)
        
        
        
        // for iIterU==iNbInnerIter-1
        for (iy=0; iy< iNy; iy++)
            for(ix=0; ix< iNx; ix++)
        {
            iX = X(ix,iy);
            iXb = Xinv(ix,iy);
            iXNeigh4 = iX*iNbNeigh4;
            iXNeigh2 = iX*iNbNeigh2;
            fSum1 = 0.0;
            fSum2 = 0.0;
            fDivWD = 0.0;
            iNbNeighX = piSY[iXb];
            for (i=0; i<iNbNeighX; i++)
            {
                ixY = piY[X4(iXNeigh4,2*i)];
                iyY = piY[X4(iXNeigh4,2*i+1)];
                iY = X(ixY,iyY);
                iYNeigh4 = iY* iNbNeigh4;
                iYNeigh2 = iY* iNbNeigh2;
                
                fw = pfW[X4(iXNeigh4,2*i)];
                fSqrtw = pfW[X4(iXNeigh4,2*i+1)];
                
                fdij = pfdNew[X4(iXNeigh2,i)];
                fbij = pfbNew[X4(iXNeigh2,i)];
                fdji = pfdNew[X4(iYNeigh2,i)];
                fbji = pfbNew[X4(iYNeigh2,i)];
                fSum1 += fw;
                fSum2 += fw* pfuNew[iY];
                fDivWD += fSqrtw*( fdij-fdji );
                fDivWD -= fSqrtw*( fbij-fbji );
                
            } // end for (i=0; i<piSY[iXb]; i++)
            fDen = fMu + 2.0*fLambda*fSum1; // 2 in graph Laplacian
            pfuNew[iX] = (2.0*fLambda*fSum2 + fMu*pfIm0[iX] - fLambda*fDivWD)/ fDen;
            
            // b
            fNormU = 0.0;
            for (i=0; i<iNbNeighX; i++)
            {
                ixY = piY[X4(iXNeigh4,2*i)];
                iyY = piY[X4(iXNeigh4,2*i+1)];
                fw = pfW[X4(iXNeigh4,2*i)];
                fSqrtw = pfW[X4(iXNeigh4,2*i+1)];
                fbij = pfbNew[X4(iXNeigh2,i)];
                iY = X(ixY,iyY);
                fGuij = fSqrtw* (pfuNew[iY]-pfuNew[iX]);
                pfbNew[X4(iXNeigh2,i)] += (fGuij - pfdNew[X4(iXNeigh2,i)]);
            } // end for (i=0; i<piSY[iXb]; i++)
            
            } // end for iy,ix
        
        
        
        // d
        for (iy=0; iy< iNy; iy++)
            for(ix=0; ix< iNx; ix++)
        {
            iX = X(ix,iy);
            iXNeigh4 = iX*iNbNeigh4;
            iXNeigh2 = iX*iNbNeigh2;
            iXb = Xinv(ix,iy);
            fNormU = 0.0;
            iNbNeighX = piSY[iXb];
            for (i=0; i<iNbNeighX; i++)
            {
                ixY = piY[X4(iXNeigh4,2*i)];
                iyY = piY[X4(iXNeigh4,2*i+1)];
                fw = pfW[X4(iXNeigh4,2*i)];
                fSqrtw = pfW[X4(iXNeigh4,2*i+1)];
                fbij = pfbNew[X4(iXNeigh2,i)];
                iY = X(ixY,iyY);
                fGuij = fSqrtw* (pfuNew[iY]-pfuNew[iX]);
                fNormU += SQR(fGuij)+ SQR(fbij);
                pfdNew[X4(iXNeigh2,i)] = fGuij+ fbij;
                pfGu[i] = fGuij;
            }
            
            fNormU = sqrtf(fNormU);
            // printf("fNormU= %.3f\n",fNormU);
            if ( fNormU<fctST )
                for (i=0; i<iNbNeighX; i++)
                    pfdNew[X4(iXNeigh2,i)] = 0.0;
            else
            {
                fTemp = fNormU-fctST; fTemp /= fNormU;
                for (i=0; i<iNbNeighX; i++)
                    pfdNew[X4(iXNeigh2,i)] *= fTemp;
            }
            }
        
        
    } // END for (iIter=0; iIter< iNbIters; iIter++)
    
    

    
    //free( (float *) pfuTemp );
    //free( (float *) pfGu );
    
    free( pfuTemp );
    free( pfGu );        
    
    
    end_time = clock();
    //printf("Computing Time for NL-TV with Split-Bregman= %.3f sec\n \n",difftime(end_time,start_time)/1000);


}
/****************************************/






/**************************************** End of file ****************************************/
