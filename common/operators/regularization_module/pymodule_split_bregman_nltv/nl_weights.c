
/***************************************************************************/
/* Name:          compute_weights_mex.c                                    */
/* Description:                                                            */
/* Date:          09-01-30                                                 */
/* Author:        Xavier Bresson (xbresson@math.ucla.edu)                  */
/*                Xiaoqun Zhang                                            */
/***************************************************************************/


/*  mex compute_fastNLWeights_mex.c */


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
#define X4b(iXinv,i) (i)*iNyx+ iXinv
#define Xe(ix,iy) (ix)*iNye+ (iy)
#define XWe(ix,iy,i1,i2) ((ix)*iNy+ iy)*isqw2+ (i1)*iw + i2
#define Xd(idx,idy) (idx+iw2)*iw+ (idy+iw2)
#define Xid1(iXsqw2,iXd) iXsqw2 + iXd
#define Xid2(iXNeigh2,i) iXNeigh2 + i
#define XWe2(iXsqw2,iXd) iXsqw2 + iXd   

                    

#define ABS(x) ( x >= 0.0 ? x : -x )
#define SQR(x) (x)*(x)

#define SWAP(a,b,tmp) tmp=a; a=b; b=tmp



static union
{
  double d;
  struct {
/* #ifdef LITTLE_ENDIAN */
    int j,i;
/* #else
//     int i,j;
// #endif*/
  } n;
} _eco;
#define EXP_A (1048576/0.69314718055994530942)
#define EXP_C 60801
#define DEXP(y) (_eco.n.i = EXP_A*(y) + (1072693248 - EXP_C), _eco.d)
#define EXP(y) (float)DEXP(y)



float SQRT(float number) {
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
//
void nl_weights( float *Img , float *param , float *pfW , int *piY , int *piSizeNeigh  )
{
    
  /* iNbOut: number of outputs
     pmxOut: array of pointers to output arguments */
    
  /* iNbIn: number of inputs
     pmxIn: array of pointers to input arguments */
    
    
    //float   *Img, *param, 
    float   *pfW2, *pfW2b, fh;
    float   *pfSdx, *Imge, fDif, fDist, *pfWe;
    float   fcurrent, fTmp;
    
    int     iNy, iNx, iNdim, iDim[3], iDisplay;
    int     iNbNeigh, iN3, im, im2, iw, iw2, ic1;
    int     iy, ix, i, iye, ixe, i2, ixt, iyt, iNbBestNeigh;
    int     iNyx, iNxe, iNye, ik, idx, idy, ip;
    int     adr, adr1, adr2, adr3, adr4, iyp1, iyp2, ixp1, ixp2, iNyxw;
    int     iNxe2, iNye2, iNyxe, iX;
    int     istart, iend, imiddle, icurrent, isqw2, iTmp, i3;
    int     *pidx, *pidy, *pidxb, *pidyb, iStartCpt, iXd, iY, iXsqw2; 
    int     iXNeigh2, iYNeigh2, iXNeigh4, iYNeigh4;
    int     iXinv, iYinv, i1a, i1b, i2a, i2b, iNbNeigh2, iNbNeigh4;
    int     iIncludeCloseNeigh, j1, j2, j3;
    int     iNbNeighToSort, iIsolatedPt;
    
    short   *psId1, *psId2, *psId3;
    
    time_t  start_time, end_time;
    time_t  start_time2, end_time2;
    
    start_time = clock();
    
    
    //printf("\nStart NL-Weights\n");
    
    
    
    //Img = mxGetData(pmxIn[0]);
    
    //param = mxGetData(pmxIn[1]);
    
    
    
    iNx = (int) param[0];
    iNy = (int) param[1];
    im = (int) param[2];
    iw = (int) param[3];
    fh = param[4];
    iNbNeigh = (int) param[5];
    iIncludeCloseNeigh = (int) param[6];
    //printf("iNy= %i, iNx= %i, im= %i, iw= %i, fh= %.3f\n",iNy,iNx,im,iw,fh);
    
    
    im2 = (im-1)/2;
    iw2 = (iw-1)/2;
    ic1 = im2+iw2;
    //printf("iNbNeigh= %i, im2= %i, iw2= %i, ic1= %i\n",iNbNeigh,im2,iw2,ic1);
    
    
    if (iNbNeigh>iw*iw-4)
    {
        iNbNeigh = iw*iw;
        iNbBestNeigh = iNbNeigh;
        iIncludeCloseNeigh = NO;
    }
    else
    {
        iNbBestNeigh = iNbNeigh;
        if ( iIncludeCloseNeigh==YES ) iNbNeigh += 4;
    }
    //printf("iNbNeigh= %i, iNbBestNeigh= %i, iIncludeCloseNeigh= %i\n",iNbNeigh,iNbBestNeigh,iIncludeCloseNeigh);

    
    iNyx = iNy* iNx;
    
    
    
    iNyxw = iNy*iNx*iw;
    
    
    iNxe = iNx + 2* iw2;
    iNye = iNy + 2* iw2;
    
    iNxe2 = 2*iNxe-2;
    iNye2 = 2*iNye-2;
    
    isqw2 = iw*iw;
    
    
    iNyxe = iNxe*iNye;
    
    iNbNeigh2 = 2*iNbNeigh;
    iNbNeigh4 = 4*iNbNeigh;
    
    
    
    
    
    
    
    
    //iN3 = iNbNeigh* 2* 2;
    //iNdim = 3;
    //iDim[0] = iNx;
    //iDim[1] = iNy;
    //iDim[2] = iN3;
    //pmxOut[0] = mxCreateNumericArray(iNdim,(const int*)iDim,mxSINGLE_CLASS,mxREAL);
    //pfW = mxGetData(pmxOut[0]);
    
    
    //iN3 =iNbNeigh* 2* 2;
    //iNdim = 3;
    //iDim[0] = iNx;
    //iDim[1] = iNy;
    //iDim[2] = iN3;
    //pmxOut[1] = mxCreateNumericArray(iNdim,(const int*)iDim,mxINT32_CLASS,mxREAL);
    //piY = mxGetData(pmxOut[1]);
    
    
    
    //iNdim = 2;
    //iDim[0] = iNx;
    //iDim[1] = iNy;
    //pmxOut[2] = mxCreateNumericArray(iNdim,(const int*)iDim,mxINT32_CLASS,mxREAL);
    //piSizeNeigh = mxGetData(pmxOut[2]);
    
    
    
    
    //iN3 =iNbNeigh* 2* 2;
    //iNdim = 3;
    //iDim[0] = iNx;
    //iDim[1] = iNy;
    //iDim[2] = iN3;
    
    //pmxOut[3] = mxCreateNumericArray(iNdim,(const int*)iDim,mxSINGLE_CLASS,mxREAL);
    //pfWmat = mxGetData(pmxOut[3]);
    
    //pmxOut[4] = mxCreateNumericArray(iNdim,(const int*)iDim,mxINT32_CLASS,mxREAL);
    //piYmat = mxGetData(pmxOut[4]);
    
    
    
    
    
    
    
    
    
    
    pfW2 = (float *) calloc( (unsigned)(iw*iw), sizeof(float) );
    if (!pfW2)
        printf("Memory allocation failure\n");
    
    pfW2b = (float *) calloc( (unsigned)(iNbNeigh), sizeof(float) );
    if (!pfW2b)
        printf("Memory allocation failure\n");
    
    pidx = (int *) calloc( (unsigned)(iw*iw), sizeof(int) );
    if (!pidx)
        printf("Memory allocation failure\n");
    
    pidy = (int *) calloc( (unsigned)(iw*iw), sizeof(int) );
    if (!pidy)
        printf("Memory allocation failure\n");
    
    pidxb = (int *) calloc( (unsigned)(iNbNeigh), sizeof(int) );
    if (!pidxb)
        printf("Memory allocation failure\n");
    
    pidyb = (int *) calloc( (unsigned)(iNbNeigh), sizeof(int) );
    if (!pidyb)
        printf("Memory allocation failure\n");
    
    psId1 = (short *) calloc( (unsigned)(iNx*iNy*iw*iw), sizeof(short) );
    if (!psId1)
        printf("Memory allocation failure\n");
     
    psId2 = (short *) calloc( (unsigned)(iNx*iNy*iNbNeigh2), sizeof(short) );
    if (!psId2)
        printf("Memory allocation failure\n");
    
    psId3 = (short *) calloc( (unsigned)(iw*iw), sizeof(short) );
    if (!psId3)
        printf("Memory allocation failure\n");
    
    pfSdx = (float *) calloc( (unsigned)(iNxe*iNye), sizeof(float) );
    if (!pfSdx)
        printf("Memory allocation failure\n");
    
    Imge = (float *) calloc( (unsigned)(iNxe*iNye), sizeof(float) );
    if (!Imge)
        printf("Memory allocation failure\n");
    
    pfWe = (float *) calloc( (unsigned)(iNx*iNy*iw*iw), sizeof(float) );
    if (!pfWe)
        printf("Memory allocation failure\n");
    
    
    
    
    // init
    start_time2 = clock();
    for (i=0; i< iNyx; i++)
    {
        j1 = i*iNbNeigh4;
        j2 = i*isqw2;
        j3 = i*iNbNeigh2;
        for (i2=0; i2< iNbNeigh4; i2++) { pfW[j1+i2] = 0.0; piY[j1+i2] = 0; }
        for (i2=0; i2< isqw2; i2++) psId1[j2+i2] = 0;
        for (i2=0; i2< iNbNeigh2; i2++) psId2[j3+i2] = 0;
        piSizeNeigh[i] = 0;
    }
    end_time2 = clock();
    //printf("Step #1 (init): Time= %.3f sec",difftime(end_time2,start_time2)/1000);
    

    
    
    
    
    
    // compute extended image
    for (ixe=0,ik=0;ixe<iNxe;ixe++)
    {
        if (ixe<iw2) ix=iw2-ixe;
        else if (ixe>iNx+iw2-1) ix=2*iNx+iw2-ixe-2;
        else ix=ixe-iw2;
        
        for (iye=0;iye<iNye;iye++,ik++)
        {
            if (iye<iw2)  iy=iw2-iye;
            else if (iye>iNy+iw2-1) iy=2*iNy+iw2-iye-2;
            else iy=iye-iw2;
            
            Imge[ik]=Img[X(ix,iy)];
        }
    }


    
    
    
 
    // compute differences between patches
    start_time2 = clock();
    for (idy=-iw2;idy<=iw2;idy++)
        for (idx=-iw2;idx<=iw2;idx++)
    {
        
        // clear image 
        for (ip=0;ip<iNyxe;ip++) pfSdx[ip]=0.0;
        ixe=0; iye=0;
        ixt=ixe+idx;
        iyt=iye+idy;
        if (ixt<0)ixt=-ixt; if (ixt>=iNxe) ixt=iNxe2-ixt;
        if (iyt<0)iyt=-iyt; if (iyt>=iNye) iyt=iNye2-iyt;
        fDif = Imge[Xe(ixe,iye)]-Imge[Xe(ixt,iyt)];
        pfSdx[Xe(ixe,iye)] = fDif*fDif;
        ixe=0;
        for (iye=1;iye<iNye;iye++)
        {
            ixt=ixe+idx;
            iyt=iye+idy;
            if (ixt<0)ixt=-ixt; if (ixt>=iNxe) ixt=iNxe2-ixt;
            if (iyt<0)iyt=-iyt; if (iyt>=iNye) iyt=iNye2-iyt;
            fDif = Imge[Xe(ixe,iye)]-Imge[Xe(ixt,iyt)];
            pfSdx[Xe(ixe,iye)] = pfSdx[Xe(ixe,iye-1)] + fDif*fDif;
        }
        
        iye=0;
        for (ixe=1;ixe<iNxe;ixe++)
        {
            ixt=ixe+idx;
            iyt=iye+idy;
            if (ixt<0)ixt=-ixt; if (ixt>=iNxe) ixt=iNxe2-ixt;
            if (iyt<0)iyt=-iyt; if (iyt>=iNye) iyt=iNye2-iyt;
            fDif = Imge[Xe(ixe,iye)]-Imge[Xe(ixt,iyt)];
            pfSdx[Xe(ixe,iye)] = pfSdx[Xe(ixe-1,iye)] + fDif*fDif;
        }
        
        for (ixe=1;ixe<iNxe;ixe++)
            for (iye=1;iye<iNye;iye++)
        {
            ixt=ixe+idx;
            iyt=iye+idy;
            if (ixt<0)ixt=-ixt; if (ixt>=iNxe) ixt=2*iNxe-ixt-2;
            if (iyt<0)iyt=-iyt; if (iyt>=iNye) iyt=2*iNye-iyt-2;
            fDif = Imge[Xe(ixe,iye)]-Imge[Xe(ixt,iyt)];
            pfSdx[Xe(ixe,iye)] = pfSdx[Xe(ixe-1,iye)] + pfSdx[Xe(ixe,iye-1)] - pfSdx[Xe(ixe-1,iye-1)] + fDif*fDif;
        }
        
        // Original Image zone
        for (iye=iw2;iye<iNye-iw2;iye++)
            for (ixe=iw2;ixe<iNxe-iw2;ixe++)       
        {
            // output image position
            iy=iye-iw2;
            ix=ixe-iw2;
            adr=X(ix,iy);
            // Sdx(iy-s,ix-s)-Sdx(iy+s,ix+s) to get the patch difference
            iyp1=iye-im2;
            ixp1=ixe-im2;
            iyp2=iye+im2;
            ixp2=ixe+im2;
        
            adr1=Xe(ixp1,iyp1); 
            adr2=Xe(ixp2,iyp2); 
            adr3=Xe(ixp1,iyp2); 
            adr4=Xe(ixp2,iyp1); 
            
            fDist = pfSdx[adr2] - pfSdx[adr3] - pfSdx[adr4] + pfSdx[adr1];
            pfWe[XWe(ix,iy,idx+iw2,idy+iw2)] = fDist;
            }
        
        }
    end_time2 = clock();
    //printf("\nStep #2 (distance between patches): Time= %.3f sec\n",difftime(end_time2,start_time2)/1000);
    
    
    
    
        
    // CENTER
    start_time2 = clock();
    for (iy=ic1; iy< iNy-ic1; iy++)
        for(ix=ic1; ix< iNx-ic1; ix++)
    {
        iX = X(ix,iy);
        iXsqw2 = iX*isqw2;
        for (idy=-iw2; idy<= iw2; idy++)
            for(idx=-iw2; idx<= iw2; idx++)
        {
                iXd = Xd(idx,idy);
                pfW2[iXd] = pfWe[XWe2(iXsqw2,iXd)];
                pidx[iXd] = idx; 
                pidy[iXd] = idy;
            }
        
        // 4 neighbors
        if ( iIncludeCloseNeigh==YES )
        {
            idx = 1; idy = 0; pfW2[Xd(idx,idy)] = 0.0;
            idx = -1; idy = 0; pfW2[Xd(idx,idy)] = 0.0;
            idx = 0; idy = 1; pfW2[Xd(idx,idy)] = 0.0;
            idx = 0; idy = -1; pfW2[Xd(idx,idy)] = 0.0;
            iNbNeighToSort = iNbNeigh;
        }
        else        
            iNbNeighToSort = iNbBestNeigh;
        
        
        for (i2=0; i2< iNbNeighToSort; i2++)  pfW2b[i2] = 1024.0;
        for (i2=0; i2< isqw2; i2++)
        {
            fcurrent = pfW2[i2];
            if ( fcurrent<pfW2b[iNbNeighToSort-1] )
            {
                // dichotomy
                istart = 0;
                iend = iNbNeighToSort-1;
                while ( iend-istart>1 )
                {
                    imiddle = (iend-istart)/2;
                    if (pfW2b[istart+imiddle] > fcurrent)
                        iend = istart+imiddle;
                    else
                        istart = istart+imiddle;
                }
                if (pfW2b[istart] > fcurrent)
                    icurrent = istart;
                else
                    icurrent = iend;
                
                // shifting
                for(i3=iNbNeighToSort-2; i3>=icurrent; i3--)
                {
                    SWAP(pfW2b[i3],pfW2b[i3+1],fTmp);
                    SWAP(pidxb[i3],pidxb[i3+1],iTmp);
                    SWAP(pidyb[i3],pidyb[i3+1],iTmp);
                }
                
                // new value
                pfW2b[icurrent] = fcurrent;
                pidxb[icurrent] = pidx[i2];
                pidyb[icurrent] = pidy[i2];
                
            }
        } // end for (i2=0; i2< isqw2; i2++)
        
        
        
        // 4 neighbors
        if ( iIncludeCloseNeigh==YES )
        {
            icurrent = 0;
            idx = 0; idy = 0; pfW2b[icurrent] = pfW2b[1+4];
            pidxb[icurrent] = idx; pidyb[icurrent] = idy; icurrent++;
            idx = 1; idy = 0; pfW2b[icurrent] = pfWe[XWe2(iXsqw2,Xd(idx,idy))];
            pidxb[icurrent] = idx; pidyb[icurrent] = idy; icurrent++;
            idx = -1; idy = 0; pfW2b[icurrent] = pfWe[XWe2(iXsqw2,Xd(idx,idy))];
            pidxb[icurrent] = idx; pidyb[icurrent] = idy; icurrent++;
            idx = 0; idy = 1; pfW2b[icurrent] = pfWe[XWe2(iXsqw2,Xd(idx,idy))];
            pidxb[icurrent] = idx; pidyb[icurrent] = idy; icurrent++;
            idx = 0; idy = -1; pfW2b[icurrent] = pfWe[XWe2(iXsqw2,Xd(idx,idy))];
            pidxb[icurrent] = idx; pidyb[icurrent] = idy;
        }
        else
            pfW2b[0] = pfW2b[1];
        
        
        // compute symmetric W
        if ( EXP(-pfW2b[0]/fh)>0.2 ) iIsolatedPt = 0; else iIsolatedPt = 1;
        iXinv = Xinv(ix,iy);
        iXNeigh2 = iX*iNbNeigh2;
        iXNeigh4 = iX*iNbNeigh4;
        iStartCpt = 0;
        for (i=0; i< iNbNeigh; i++)
        {
            idx = pidxb[i];
            idy = pidyb[i];
            iXd = Xd(idx,idy);
            iYinv = Xinv(ix+idx,iy+idy);
            
            if ( psId1[Xid1(iXsqw2,iXd)]==0 )
            {
                iY = X(ix+idx,iy+idy);
                iYNeigh2 = iY* iNbNeigh2;
                iYNeigh4 = iY* iNbNeigh4;
                for (i2=iStartCpt; i2< iNbNeigh2; i2++)
                    if ( psId2[Xid2(iXNeigh2,i2)]==0 && psId2[Xid2(iYNeigh2,i2)]==0 )
                {
                    i1a = X4(iXNeigh4,2*i2);
                    i1b = i1a + 1;
                    i2a = X4(iYNeigh4,2*i2);
                    i2b = i2a + 1;
                    
                    if (iIsolatedPt==0) pfW[i1a] = EXP(-pfW2b[i]/fh); else pfW[i1a] = 0.2;
                    pfW[i1b] = sqrtf(pfW[i1a]);
                    pfW[i2a] = pfW[i1a];
                    pfW[i2b] = pfW[i1b];
                    piY[i1a] = ix+idx;
                    piY[i1b] = iy+idy;
                    piY[i2a] = ix;
                    piY[i2b] = iy;

                    psId2[Xid2(iXNeigh2,i2)] = 1;
                    psId2[Xid2(iYNeigh2,i2)] = 1;
                    psId1[Xid1(iXsqw2,iXd)] = 1;
                    psId1[Xid1(iY*isqw2,Xd(-idx,-idy))] = 1;
                    if ( i2==iStartCpt+1 ) iStartCpt++;
                    
                    if ( piSizeNeigh[iXinv]<i2 ) piSizeNeigh[iXinv]=i2;
                    if ( piSizeNeigh[iYinv]<i2 ) piSizeNeigh[iYinv]=i2;
                    
                    i2 = iNbNeigh2;
                    }  
                
            } // end if ( piId1[Xid1(iX,iXd)]==0 )
            
        }// end for (i=0; i< iNbNeigh; i++)
        
        
        }// END
    end_time2 = clock();
    //printf("Step #3 (sort m best values): Time= %.3f sec\n",difftime(end_time2,start_time2)/1000);
   
    
    
    
    
    
    
    
    
    
    // BORDER #1
    start_time2 = clock();
    for (iy=0; iy< ic1; iy++)
        for(ix=0; ix< iNx; ix++)
    {
        iX = X(ix,iy);
        iXsqw2 = iX*isqw2;
        for (idy=-iw2; idy<= iw2; idy++)
            for(idx=-iw2; idx<= iw2; idx++)
                if ( ix+idx>=0 && ix+idx<iNx && iy+idy>=0 && iy+idy<iNy )
        {
            iXd = Xd(idx,idy);
            pfW2[iXd] = pfWe[XWe2(iXsqw2,iXd)];
            pidx[iXd] = idx;
            pidy[iXd] = idy;
            psId3[iXd] = 0;
                }
                else
        {
            iXd = Xd(idx,idy);
            pfW2[Xd(idx,idy)] = 1024.0;
            pidx[iXd] = idx;
            pidy[iXd] = idy;
            psId3[iXd] = 1;
                }
        
        
        // 4 neighbors
        if ( iIncludeCloseNeigh==YES )
        {
            idx = 1; idy = 0; pfW2[Xd(idx,idy)] = 0.0;
            idx = -1; idy = 0; pfW2[Xd(idx,idy)] = 0.0;
            idx = 0; idy = 1; pfW2[Xd(idx,idy)] = 0.0;
            idx = 0; idy = -1; pfW2[Xd(idx,idy)] = 0.0;
            iNbNeighToSort = iNbNeigh;
        }
        else        
            iNbNeighToSort = iNbBestNeigh;
        
        
        for (i2=0; i2< iNbNeighToSort; i2++)  pfW2b[i2] = 1024.0;
        for (i2=0; i2< isqw2; i2++)
        {
            fcurrent = pfW2[i2];
            if ( fcurrent<pfW2b[iNbNeighToSort-1] )
            {
                // dichotomy
                istart = 0;
                iend = iNbNeighToSort-1;
                while ( iend-istart>1 )
                {
                    imiddle = (iend-istart)/2;
                    if (pfW2b[istart+imiddle] > fcurrent)
                        iend = istart+imiddle;
                    else
                        istart = istart+imiddle;
                }
                if (pfW2b[istart] > fcurrent)
                    icurrent = istart;
                else
                    icurrent = iend;
                
                // shifting
                for(i3=iNbNeighToSort-2; i3>=icurrent; i3--)
                {
                    SWAP(pfW2b[i3],pfW2b[i3+1],fTmp);
                    SWAP(pidxb[i3],pidxb[i3+1],iTmp);
                    SWAP(pidyb[i3],pidyb[i3+1],iTmp);
                }
                
                // new value
                pfW2b[icurrent] = fcurrent;
                pidxb[icurrent] = pidx[i2];
                pidyb[icurrent] = pidy[i2];
                
            }
        } // end for (i2=0; i2< isqw2; i2++)
        
        
        // 4 neighbors
        if ( iIncludeCloseNeigh==YES )
        {
            icurrent = 0;
            idx = 0; idy = 0; pfW2b[icurrent] = pfW2b[1+4];
            pidxb[icurrent] = idx; pidyb[icurrent] = idy; icurrent++;
            idx = 1; idy = 0;
            if ( ix+idx>=0 && ix+idx<iNx && iy+idy>=0 && iy+idy<iNy )
            { pfW2b[icurrent] = pfWe[XWe2(iXsqw2,Xd(idx,idy))]; pidxb[icurrent] = idx; pidyb[icurrent] = idy; icurrent++;}
            idx = -1; idy = 0;
            if ( ix+idx>=0 && ix+idx<iNx && iy+idy>=0 && iy+idy<iNy )
            { pfW2b[icurrent] = pfWe[XWe2(iXsqw2,Xd(idx,idy))]; pidxb[icurrent] = idx; pidyb[icurrent] = idy; icurrent++;}
            idx = 0; idy = 1;
            if ( ix+idx>=0 && ix+idx<iNx && iy+idy>=0 && iy+idy<iNy )
            { pfW2b[icurrent] = pfWe[XWe2(iXsqw2,Xd(idx,idy))]; pidxb[icurrent] = idx; pidyb[icurrent] = idy; icurrent++;}
            idx = 0; idy = -1;
            if ( ix+idx>=0 && ix+idx<iNx && iy+idy>=0 && iy+idy<iNy )
            { pfW2b[icurrent] = pfWe[XWe2(iXsqw2,Xd(idx,idy))]; pidxb[icurrent] = idx; pidyb[icurrent] = idy;}
        }
        else
            pfW2b[0] = pfW2b[1];
        
        // compute symmetric W
        if ( EXP(-pfW2b[0]/fh)>0.2 ) iIsolatedPt = 0; else iIsolatedPt = 1;
        iXinv = Xinv(ix,iy);
        iXNeigh2 = iX*iNbNeigh2;
        iXNeigh4 = iX*iNbNeigh4;
        iStartCpt = 0;
        for (i=0; i< iNbNeigh; i++)
        {
            idx = pidxb[i];
            idy = pidyb[i];
            iXd = Xd(idx,idy);
            if ( psId3[iXd]==0 )
            {
                iYinv = Xinv(ix+idx,iy+idy);
                if ( psId1[Xid1(iXsqw2,iXd)]==0 )
                {
                    iY = X(ix+idx,iy+idy);
                    iYNeigh2 = iY* iNbNeigh2;
                    iYNeigh4 = iY* iNbNeigh4;
                    for (i2=iStartCpt; i2< iNbNeigh2; i2++)
                        if ( psId2[Xid2(iXNeigh2,i2)]==0 && psId2[Xid2(iYNeigh2,i2)]==0 )
                    {
                        i1a = X4(iXNeigh4,2*i2);
                        i1b = i1a + 1;
                        i2a = X4(iYNeigh4,2*i2);
                        i2b = i2a + 1;
                        if (iIsolatedPt==0) pfW[i1a] = EXP(-pfW2b[i]/fh); else pfW[i1a] = 0.2;
                        pfW[i1b] = sqrtf(pfW[i1a]);
                        pfW[i2a] = pfW[i1a];
                        pfW[i2b] = pfW[i1b];
                        piY[i1a] = ix+idx;
                        piY[i1b] = iy+idy;
                        piY[i2a] = ix;
                        piY[i2b] = iy;
                        psId2[Xid2(iXNeigh2,i2)] = 1;
                        psId2[Xid2(iYNeigh2,i2)] = 1;
                        psId1[Xid1(iXsqw2,iXd)] = 1;
                        psId1[Xid1(iY*isqw2,Xd(-idx,-idy))] = 1;
                        if ( i2==iStartCpt+1 ) iStartCpt++;
                        if ( piSizeNeigh[iXinv]<i2 ) piSizeNeigh[iXinv]=i2;
                        if ( piSizeNeigh[iYinv]<i2 ) piSizeNeigh[iYinv]=i2;
                        i2 = iNbNeigh2;
                        }
                } // end if ( piId1[Xid1(iX,iXd)]==0 )
            } // end for (i=0; i< iNbNeigh; i++)
        }
        } // END
    

    //printf("HERE !!!\n");
    
    
    //printf("HERE 3d\n");
    //printf("ix= %i, iy = %i, iYx= %i, iYy = %i, i= %i, i2= %i, iNbNeighBorder= %i\n",ix,iy,ix+idx,iy+idy,i,i2,iNbNeighBorder);
                 
    
    
    
    
    
    
    
    
    
    
    // BORDER #2
    for (iy=iNy-ic1; iy< iNy; iy++)
        for(ix=0; ix< iNx; ix++)
    {
        iX = X(ix,iy);
        iXsqw2 = iX*isqw2;
        for (idy=-iw2; idy<= iw2; idy++)
            for(idx=-iw2; idx<= iw2; idx++)
                if ( ix+idx>=0 && ix+idx<iNx && iy+idy>=0 && iy+idy<iNy )
        {
            iXd = Xd(idx,idy);
            pfW2[iXd] = pfWe[XWe2(iXsqw2,iXd)];
            pidx[iXd] = idx;
            pidy[iXd] = idy;
            psId3[iXd] = 0;
                }
                else
        {
            iXd = Xd(idx,idy);
            pfW2[Xd(idx,idy)] = 1024.0;
            pidx[iXd] = idx;
            pidy[iXd] = idy;
            psId3[iXd] = 1;
                }
        
        
        // 4 neighbors
        if ( iIncludeCloseNeigh==YES )
        {
            idx = 1; idy = 0; pfW2[Xd(idx,idy)] = 0.0;
            idx = -1; idy = 0; pfW2[Xd(idx,idy)] = 0.0;
            idx = 0; idy = 1; pfW2[Xd(idx,idy)] = 0.0;
            idx = 0; idy = -1; pfW2[Xd(idx,idy)] = 0.0;
            iNbNeighToSort = iNbNeigh;
        }
        else        
            iNbNeighToSort = iNbBestNeigh;
        
        
        for (i2=0; i2< iNbNeighToSort; i2++)  pfW2b[i2] = 1024.0;
        for (i2=0; i2< isqw2; i2++)
        {
            fcurrent = pfW2[i2];
            if ( fcurrent<pfW2b[iNbNeighToSort-1] )
            {
                // dichotomy
                istart = 0;
                iend = iNbNeighToSort-1;
                while ( iend-istart>1 )
                {
                    imiddle = (iend-istart)/2;
                    if (pfW2b[istart+imiddle] > fcurrent)
                        iend = istart+imiddle;
                    else
                        istart = istart+imiddle;
                }
                if (pfW2b[istart] > fcurrent)
                    icurrent = istart;
                else
                    icurrent = iend;
                
                // shifting 
                for(i3=iNbNeighToSort-2; i3>=icurrent; i3--)
                {
                    SWAP(pfW2b[i3],pfW2b[i3+1],fTmp);
                    SWAP(pidxb[i3],pidxb[i3+1],iTmp);
                    SWAP(pidyb[i3],pidyb[i3+1],iTmp);
                }
                
                // new value
                pfW2b[icurrent] = fcurrent;
                pidxb[icurrent] = pidx[i2];
                pidyb[icurrent] = pidy[i2];
                
            }
        } // end for (i2=0; i2< isqw2; i2++)
        
        
        // 4 neighbors
        if ( iIncludeCloseNeigh==YES )
        {
            icurrent = 0;
            idx = 0; idy = 0; pfW2b[icurrent] = pfW2b[1+4];
            pidxb[icurrent] = idx; pidyb[icurrent] = idy; icurrent++;
            idx = 1; idy = 0;
            if ( ix+idx>=0 && ix+idx<iNx && iy+idy>=0 && iy+idy<iNy )
            { pfW2b[icurrent] = pfWe[XWe2(iXsqw2,Xd(idx,idy))]; pidxb[icurrent] = idx; pidyb[icurrent] = idy; icurrent++;}
            idx = -1; idy = 0;
            if ( ix+idx>=0 && ix+idx<iNx && iy+idy>=0 && iy+idy<iNy )
            { pfW2b[icurrent] = pfWe[XWe2(iXsqw2,Xd(idx,idy))]; pidxb[icurrent] = idx; pidyb[icurrent] = idy; icurrent++;}
            idx = 0; idy = 1;
            if ( ix+idx>=0 && ix+idx<iNx && iy+idy>=0 && iy+idy<iNy )
            { pfW2b[icurrent] = pfWe[XWe2(iXsqw2,Xd(idx,idy))]; pidxb[icurrent] = idx; pidyb[icurrent] = idy; icurrent++;}
            idx = 0; idy = -1;
            if ( ix+idx>=0 && ix+idx<iNx && iy+idy>=0 && iy+idy<iNy )
            { pfW2b[icurrent] = pfWe[XWe2(iXsqw2,Xd(idx,idy))]; pidxb[icurrent] = idx; pidyb[icurrent] = idy;}
            else
            { pfW2b[icurrent] = 1024.0; pidxb[icurrent] = 0; pidyb[icurrent] = 0;}
        }
        else
            pfW2b[0] = pfW2b[1];
        
        // compute symmetric W
        if ( EXP(-pfW2b[0]/fh)>0.2 ) iIsolatedPt = 0; else iIsolatedPt = 1;
        iXinv = Xinv(ix,iy);
        iXNeigh2 = iX*iNbNeigh2;
        iXNeigh4 = iX*iNbNeigh4;
        iStartCpt = 0;
        for (i=0; i< iNbNeigh; i++)
        {
            idx = pidxb[i];
            idy = pidyb[i];
            iXd = Xd(idx,idy);
            if ( psId3[iXd]==0 )
            {
                iYinv = Xinv(ix+idx,iy+idy);
                if ( psId1[Xid1(iXsqw2,iXd)]==0 )
                {
                    iY = X(ix+idx,iy+idy);
                    iYNeigh2 = iY* iNbNeigh2;
                    iYNeigh4 = iY* iNbNeigh4;
                    for (i2=iStartCpt; i2< iNbNeigh2; i2++)
                        if ( psId2[Xid2(iXNeigh2,i2)]==0 && psId2[Xid2(iYNeigh2,i2)]==0 )
                    {
                        i1a = X4(iXNeigh4,2*i2);
                        i1b = i1a + 1;
                        i2a = X4(iYNeigh4,2*i2);
                        i2b = i2a + 1;
                        if (iIsolatedPt==0) pfW[i1a] = EXP(-pfW2b[i]/fh); else pfW[i1a] = 0.2;
                        pfW[i1b] = sqrtf(pfW[i1a]);
                        pfW[i2a] = pfW[i1a];
                        pfW[i2b] = pfW[i1b];
                        piY[i1a] = ix+idx;
                        piY[i1b] = iy+idy;
                        piY[i2a] = ix;
                        piY[i2b] = iy;
                        psId2[Xid2(iXNeigh2,i2)] = 1;
                        psId2[Xid2(iYNeigh2,i2)] = 1;
                        psId1[Xid1(iXsqw2,iXd)] = 1;
                        psId1[Xid1(iY*isqw2,Xd(-idx,-idy))] = 1;
                        if ( i2==iStartCpt+1 ) iStartCpt++;
                        if ( piSizeNeigh[iXinv]<i2 ) piSizeNeigh[iXinv]=i2;
                        if ( piSizeNeigh[iYinv]<i2 ) piSizeNeigh[iYinv]=i2;
                        i2 = iNbNeigh2;
                        }
                } // end if ( piId1[Xid1(iX,iXd)]==0 )
            } // end for (i=0; i< iNbNeigh; i++)
        }
        } // END
    
    
    // BORDER #3
    for (iy=ic1; iy< iNy-ic1; iy++)
        for(ix=0; ix< ic1; ix++)
    { 
        iX = X(ix,iy);
        iXsqw2 = iX*isqw2;
        for (idy=-iw2; idy<= iw2; idy++)
            for(idx=-iw2; idx<= iw2; idx++)
                if ( ix+idx>=0 && ix+idx<iNx && iy+idy>=0 && iy+idy<iNy )
        {
            iXd = Xd(idx,idy);
            pfW2[iXd] = pfWe[XWe2(iXsqw2,iXd)];
            pidx[iXd] = idx;
            pidy[iXd] = idy;
            psId3[iXd] = 0;
                }
                else
        {
            iXd = Xd(idx,idy);
            pfW2[Xd(idx,idy)] = 1024.0;
            pidx[iXd] = idx;
            pidy[iXd] = idy;
            psId3[iXd] = 1;
                }
        
        
        // 4 neighbors
        if ( iIncludeCloseNeigh==YES )
        {
            idx = 1; idy = 0; pfW2[Xd(idx,idy)] = 0.0;
            idx = -1; idy = 0; pfW2[Xd(idx,idy)] = 0.0;
            idx = 0; idy = 1; pfW2[Xd(idx,idy)] = 0.0;
            idx = 0; idy = -1; pfW2[Xd(idx,idy)] = 0.0;
            iNbNeighToSort = iNbNeigh;
        }
        else        
            iNbNeighToSort = iNbBestNeigh;
        
        
        for (i2=0; i2< iNbNeighToSort; i2++)  pfW2b[i2] = 1024.0;
        for (i2=0; i2< isqw2; i2++)
        {
            fcurrent = pfW2[i2];
            if ( fcurrent<pfW2b[iNbNeighToSort-1] )
            {
                // dichotomy
                istart = 0;
                iend = iNbNeighToSort-1;
                while ( iend-istart>1 )
                {
                    imiddle = (iend-istart)/2;
                    if (pfW2b[istart+imiddle] > fcurrent)
                        iend = istart+imiddle;
                    else
                        istart = istart+imiddle;
                }
                if (pfW2b[istart] > fcurrent)
                    icurrent = istart;
                else
                    icurrent = iend;
                
                // shifting
                for(i3=iNbNeighToSort-2; i3>=icurrent; i3--)
                {
                    SWAP(pfW2b[i3],pfW2b[i3+1],fTmp);
                    SWAP(pidxb[i3],pidxb[i3+1],iTmp);
                    SWAP(pidyb[i3],pidyb[i3+1],iTmp);
                }
                
                // new value
                pfW2b[icurrent] = fcurrent;
                pidxb[icurrent] = pidx[i2];
                pidyb[icurrent] = pidy[i2];
                
            }
        } // end for (i2=0; i2< isqw2; i2++)
        
        
        // 4 neighbors
        if ( iIncludeCloseNeigh==YES )
        {
            icurrent = 0;
            idx = 0; idy = 0; pfW2b[icurrent] = pfW2b[1+4];
            pidxb[icurrent] = idx; pidyb[icurrent] = idy; icurrent++;
            idx = 1; idy = 0;
            if ( ix+idx>=0 && ix+idx<iNx && iy+idy>=0 && iy+idy<iNy )
            { pfW2b[icurrent] = pfWe[XWe2(iXsqw2,Xd(idx,idy))]; pidxb[icurrent] = idx; pidyb[icurrent] = idy; icurrent++;}
            idx = -1; idy = 0;
            if ( ix+idx>=0 && ix+idx<iNx && iy+idy>=0 && iy+idy<iNy )
            { pfW2b[icurrent] = pfWe[XWe2(iXsqw2,Xd(idx,idy))]; pidxb[icurrent] = idx; pidyb[icurrent] = idy; icurrent++;}
            idx = 0; idy = 1;
            if ( ix+idx>=0 && ix+idx<iNx && iy+idy>=0 && iy+idy<iNy )
            { pfW2b[icurrent] = pfWe[XWe2(iXsqw2,Xd(idx,idy))]; pidxb[icurrent] = idx; pidyb[icurrent] = idy; icurrent++;}
            idx = 0; idy = -1;
            if ( ix+idx>=0 && ix+idx<iNx && iy+idy>=0 && iy+idy<iNy )
            { pfW2b[icurrent] = pfWe[XWe2(iXsqw2,Xd(idx,idy))]; pidxb[icurrent] = idx; pidyb[icurrent] = idy;}
        }
        else
            pfW2b[0] = pfW2b[1];
        
        // compute symmetric W
        if ( EXP(-pfW2b[0]/fh)>0.2 ) iIsolatedPt = 0; else iIsolatedPt = 1;
        iXinv = Xinv(ix,iy);
        iXNeigh2 = iX*iNbNeigh2;
        iXNeigh4 = iX*iNbNeigh4;
        iStartCpt = 0;
        for (i=0; i< iNbNeigh; i++)
        {
            idx = pidxb[i];
            idy = pidyb[i];
            iXd = Xd(idx,idy);
            if ( psId3[iXd]==0 )
            {
                iYinv = Xinv(ix+idx,iy+idy);
                if ( psId1[Xid1(iXsqw2,iXd)]==0 )
                {
                    iY = X(ix+idx,iy+idy);
                    iYNeigh2 = iY* iNbNeigh2;
                    iYNeigh4 = iY* iNbNeigh4;
                    for (i2=iStartCpt; i2< iNbNeigh2; i2++)
                        if ( psId2[Xid2(iXNeigh2,i2)]==0 && psId2[Xid2(iYNeigh2,i2)]==0 )
                    {
                        i1a = X4(iXNeigh4,2*i2);
                        i1b = i1a + 1;
                        i2a = X4(iYNeigh4,2*i2);
                        i2b = i2a + 1;
                        if (iIsolatedPt==0) pfW[i1a] = EXP(-pfW2b[i]/fh); else pfW[i1a] = 0.2;
                        pfW[i1b] = sqrtf(pfW[i1a]);
                        pfW[i2a] = pfW[i1a];
                        pfW[i2b] = pfW[i1b];
                        piY[i1a] = ix+idx;
                        piY[i1b] = iy+idy;
                        piY[i2a] = ix;
                        piY[i2b] = iy;
                        psId2[Xid2(iXNeigh2,i2)] = 1;
                        psId2[Xid2(iYNeigh2,i2)] = 1;
                        psId1[Xid1(iXsqw2,iXd)] = 1;
                        psId1[Xid1(iY*isqw2,Xd(-idx,-idy))] = 1;
                        if ( i2==iStartCpt+1 ) iStartCpt++;
                        if ( piSizeNeigh[iXinv]<i2 ) piSizeNeigh[iXinv]=i2;
                        if ( piSizeNeigh[iYinv]<i2 ) piSizeNeigh[iYinv]=i2;
                        i2 = iNbNeigh2;
                        }
                } // end if ( piId1[Xid1(iX,iXd)]==0 )
            } // end for (i=0; i< iNbNeigh; i++)
        }
        } // END 
    
    
    // BORDER #4
    for (iy=ic1; iy< iNy-ic1; iy++)
        for(ix=iNx-ic1; ix< iNx; ix++)
    { 
        iX = X(ix,iy);
        iXsqw2 = iX*isqw2;
        for (idy=-iw2; idy<= iw2; idy++)
            for(idx=-iw2; idx<= iw2; idx++)
                if ( ix+idx>=0 && ix+idx<iNx && iy+idy>=0 && iy+idy<iNy )
        {
            iXd = Xd(idx,idy);
            pfW2[iXd] = pfWe[XWe2(iXsqw2,iXd)];
            pidx[iXd] = idx;
            pidy[iXd] = idy;
            psId3[iXd] = 0;
                }
                else
        {
            iXd = Xd(idx,idy);
            pfW2[Xd(idx,idy)] = 1024.0;
            pidx[iXd] = idx;
            pidy[iXd] = idy;
            psId3[iXd] = 1;
                }
        
        
        // 4 neighbors
        if ( iIncludeCloseNeigh==YES )
        {
            idx = 1; idy = 0; pfW2[Xd(idx,idy)] = 0.0;
            idx = -1; idy = 0; pfW2[Xd(idx,idy)] = 0.0;
            idx = 0; idy = 1; pfW2[Xd(idx,idy)] = 0.0;
            idx = 0; idy = -1; pfW2[Xd(idx,idy)] = 0.0;
            iNbNeighToSort = iNbNeigh;
        }
        else        
            iNbNeighToSort = iNbBestNeigh;
        
        
        for (i2=0; i2< iNbNeighToSort; i2++)  pfW2b[i2] = 1024.0;
        for (i2=0; i2< isqw2; i2++)
        {
            fcurrent = pfW2[i2];
            if ( fcurrent<pfW2b[iNbNeighToSort-1] )
            {
                // dichotomy
                istart = 0;
                iend = iNbNeighToSort-1;
                while ( iend-istart>1 )
                {
                    imiddle = (iend-istart)/2;
                    if (pfW2b[istart+imiddle] > fcurrent)
                        iend = istart+imiddle;
                    else
                        istart = istart+imiddle;
                }
                if (pfW2b[istart] > fcurrent)
                    icurrent = istart;
                else
                    icurrent = iend;
                
                // shifting
                for(i3=iNbNeighToSort-2; i3>=icurrent; i3--)
                {
                    SWAP(pfW2b[i3],pfW2b[i3+1],fTmp);
                    SWAP(pidxb[i3],pidxb[i3+1],iTmp);
                    SWAP(pidyb[i3],pidyb[i3+1],iTmp);
                }
                
                // new value
                pfW2b[icurrent] = fcurrent;
                pidxb[icurrent] = pidx[i2];
                pidyb[icurrent] = pidy[i2];
                
            }
        } // end for (i2=0; i2< isqw2; i2++)
        
        
        // 4 neighbors
        if ( iIncludeCloseNeigh==YES )
        {
            icurrent = 0;
            idx = 0; idy = 0; pfW2b[icurrent] = pfW2b[1+4];
            pidxb[icurrent] = idx; pidyb[icurrent] = idy; icurrent++;
            idx = 1; idy = 0;
            if ( ix+idx>=0 && ix+idx<iNx && iy+idy>=0 && iy+idy<iNy )
            { pfW2b[icurrent] = pfWe[XWe2(iXsqw2,Xd(idx,idy))]; pidxb[icurrent] = idx; pidyb[icurrent] = idy; icurrent++;}
            idx = -1; idy = 0;
            if ( ix+idx>=0 && ix+idx<iNx && iy+idy>=0 && iy+idy<iNy )
            { pfW2b[icurrent] = pfWe[XWe2(iXsqw2,Xd(idx,idy))]; pidxb[icurrent] = idx; pidyb[icurrent] = idy; icurrent++;}
            idx = 0; idy = 1;
            if ( ix+idx>=0 && ix+idx<iNx && iy+idy>=0 && iy+idy<iNy )
            { pfW2b[icurrent] = pfWe[XWe2(iXsqw2,Xd(idx,idy))]; pidxb[icurrent] = idx; pidyb[icurrent] = idy; icurrent++;}
            idx = 0; idy = -1;
            if ( ix+idx>=0 && ix+idx<iNx && iy+idy>=0 && iy+idy<iNy )
            { pfW2b[icurrent] = pfWe[XWe2(iXsqw2,Xd(idx,idy))]; pidxb[icurrent] = idx; pidyb[icurrent] = idy;}
            else
            { pfW2b[icurrent] = 1024.0; pidxb[icurrent] = 0; pidyb[icurrent] = 0;}
        }
        else
            pfW2b[0] = pfW2b[1];
        
        // compute symmetric W
        if ( EXP(-pfW2b[0]/fh)>0.2 ) iIsolatedPt = 0; else iIsolatedPt = 1;
        iXinv = Xinv(ix,iy);
        iXNeigh2 = iX*iNbNeigh2;
        iXNeigh4 = iX*iNbNeigh4;
        iStartCpt = 0;
        for (i=0; i< iNbNeigh; i++)
        {
            idx = pidxb[i];
            idy = pidyb[i];
            iXd = Xd(idx,idy);
            if ( psId3[iXd]==0 )
            {
                iYinv = Xinv(ix+idx,iy+idy);
                if ( psId1[Xid1(iXsqw2,iXd)]==0 )
                {
                    iY = X(ix+idx,iy+idy);
                    iYNeigh2 = iY* iNbNeigh2;
                    iYNeigh4 = iY* iNbNeigh4;
                    for (i2=iStartCpt; i2< iNbNeigh2; i2++)
                        if ( psId2[Xid2(iXNeigh2,i2)]==0 && psId2[Xid2(iYNeigh2,i2)]==0 )
                    {
                        i1a = X4(iXNeigh4,2*i2);
                        i1b = i1a + 1;
                        i2a = X4(iYNeigh4,2*i2);
                        i2b = i2a + 1;
                        if (iIsolatedPt==0) pfW[i1a] = EXP(-pfW2b[i]/fh); else pfW[i1a] = 0.2;
                        pfW[i1b] = sqrtf(pfW[i1a]);
                        pfW[i2a] = pfW[i1a];
                        pfW[i2b] = pfW[i1b];
                        piY[i1a] = ix+idx;
                        piY[i1b] = iy+idy;
                        piY[i2a] = ix;
                        piY[i2b] = iy;
                        psId2[Xid2(iXNeigh2,i2)] = 1;
                        psId2[Xid2(iYNeigh2,i2)] = 1;
                        psId1[Xid1(iXsqw2,iXd)] = 1;
                        psId1[Xid1(iY*isqw2,Xd(-idx,-idy))] = 1;
                        if ( i2==iStartCpt+1 ) iStartCpt++;
                        if ( piSizeNeigh[iXinv]<i2 ) piSizeNeigh[iXinv]=i2;
                        if ( piSizeNeigh[iYinv]<i2 ) piSizeNeigh[iYinv]=i2;
                        i2 = iNbNeigh2;
                        }
                } // end if ( piId1[Xid1(iX,iXd)]==0 )
            } // end for (i=0; i< iNbNeigh; i++)
        }
        } // END
    end_time2 = clock();
    //printf("Step #4 (compute along border): Time= %.3f sec\n",difftime(end_time2,start_time2)/1000);

    
    
    
    
    // add 1 to matrix of the number of neighbors
    for (i=0; i< iNyx; i++) piSizeNeigh[i]++;
    
    
    
    
   
    
    /*      
    free( (float *) pfW2 );
    free( (float *) pfW2b );
    free( (int *) pidx );
    free( (int *) pidy );
    free( (int *) pidxb );
    free( (int *) pidyb );
    free( (short *) psId1 );
    free( (short *) psId2 );
    free( (float *) pfSdx );
    free( (float *) Imge );
    free( (float *) pfWe );
    */
    free( pfW2 );
    free( pfW2b );
    free( pidx );
    free( pidy );
    free( pidxb );
    free( pidyb );
    free( psId1 );
    free( psId2 );
    free( pfSdx );
    free( Imge );
    free( pfWe );      

    end_time = clock();
    //printf("Total computing Time for NL-Weights= %.3f sec\n \n",difftime(end_time,start_time)/1000);
  
}
/****************************************/






/**************************************** End of file ****************************************/
