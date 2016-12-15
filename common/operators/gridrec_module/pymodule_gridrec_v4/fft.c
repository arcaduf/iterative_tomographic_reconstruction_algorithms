#define SWAP(a,b) tempr=(a);(a)=(b);(b)=tempr
#define PI 3.141592653589793


/*******************************************************************/
/*******************************************************************/ 
/*
 * FFT & IFFT 1D
 */
void four1(float data[], unsigned long nn, int isign)
{
  unsigned long n,mmax,m,j,istep,i;
  float wtemp,wr,wpr,wpi,wi,theta;
  float tempr,tempi;
  
  n=nn << 1;
  
  j=1;
  for (i=1;i<n;i+=2) {
    if (j > i) {
      SWAP(data[j],data[i]);
      SWAP(data[j+1],data[i+1]);
    }
    m=n >> 1;
    while (m >= 2 && j > m) {
      j -= m;
      m >>= 1;
    }
    j += m;
  }
  mmax=2;
  while (n > mmax) {
    istep=mmax << 1;
    theta=isign*(6.28318530717959/mmax);
    wtemp=sin(0.5*theta);
    wpr = -2.0*wtemp*wtemp;
    wpi=sin(theta);
    wr=1.0;
    wi=0.0;
    for (m=1;m<mmax;m+=2) {
      for (i=m;i<=n;i+=istep) {
	j=i+mmax;
	tempr=wr*data[j]-wi*data[j+1];
	tempi=wr*data[j+1]+wi*data[j];
	data[j]=data[i]-tempr;
	data[j+1]=data[i+1]-tempi;
	data[i] += tempr;
	data[i+1] += tempi;
      }
      wr=(wtemp=wr)*wpr-wi*wpi+wr;
      wi=wi*wpr+wtemp*wpi+wi;
    }
    mmax=istep;
  }
  
}


void myFour1(float data[], unsigned long nn, int isign){

  int i;

  // Normalization of FFT and IFFT in such way that
  // one operator is the adjoint of the other one.
  // 1/sqrt(number_elements)  --->  for the FFT
  // sqrt(number_elements)    --->  for the IFFT
  // But, since the implementation in C of the IFFT,
  // is the matrhematical IFFT multiplied for the
  // number_elements the norm_factor is again
  // 1/sqrt(number_elements) as it is for the FFT
  //    Added by F. Arcadu on the 13/11/2013
  float norm_factor = 1.0/sqrt(nn); 
                                                     	

  four1(data-1,nn,isign);

  //  Normalization
  for(i=0;i<2*nn;i++)
    data[i] *= norm_factor; 
}



/*******************************************************************/
/*******************************************************************/ 
/*
 * FFT & IFFT nD
 */
void fourn( float data[] , unsigned long nn[] , int ndim , int isign)
{
  int idim;
  unsigned long i1,i2,i3,i2rev,i3rev,ip1,ip2,ip3,ifp1,ifp2;
  unsigned long ibit,k1,k2,n,nprev,nrem,ntot;
  float tempi,tempr;
  float theta,wi,wpi,wpr,wr,wtemp;
  
  for (ntot=1,idim=1;idim<=ndim;idim++)
    ntot *= nn[idim];
  
  nprev=1;
  for (idim=ndim;idim>=1;idim--) {
    n=nn[idim];
    nrem=ntot/(n*nprev);
    ip1=nprev << 1;
    ip2=ip1*n;
    ip3=ip2*nrem;
    i2rev=1;
    for (i2=1;i2<=ip2;i2+=ip1) {
      if (i2 < i2rev) {
	for (i1=i2;i1<=i2+ip1-2;i1+=2) {
	  for (i3=i1;i3<=ip3;i3+=ip2) {
	    i3rev=i2rev+i3-i2;
	    SWAP(data[i3],data[i3rev]);
	    SWAP(data[i3+1],data[i3rev+1]);
	  }
	}
      }
      ibit=ip2 >> 1;
      while (ibit >= ip1 && i2rev > ibit) {
	i2rev -= ibit;
	ibit >>= 1;
      }
      i2rev += ibit;
    }
    ifp1=ip1;
    while (ifp1 < ip2) {
      ifp2=ifp1 << 1;
      theta=isign*6.28318530717959/(ifp2/ip1);
      wtemp=sin(0.5*theta);
      wpr = -2.0*wtemp*wtemp;
      wpi=sin(theta);
      wr=1.0;
      wi=0.0;
      for (i3=1;i3<=ifp1;i3+=ip1) {
	for (i1=i3;i1<=i3+ip1-2;i1+=2) {
	  for (i2=i1;i2<=ip3;i2+=ifp2) {
	    k1=i2;
	    k2=k1+ifp1;
	    tempr=(float)wr*data[k2]-(float)wi*data[k2+1];
	    tempi=(float)wr*data[k2+1]+(float)wi*data[k2];
	    data[k2]=data[k1]-tempr;
	    data[k2+1]=data[k1+1]-tempi;
	    data[k1] += tempr;
	    data[k1+1] += tempi;
	  }
	}
	wr=(wtemp=wr)*wpr-wi*wpi+wr;
	wi=wi*wpr+wtemp*wpi+wi;
      }
      ifp1=ifp2;
    }
    nprev *= n;
  }
}



void myFour2( float *data , unsigned long nc , int isign ) {

  int ndim = 2;
  unsigned long i,j,ntot,idim,ncD;
  unsigned long nn[2];
  float norm_factor;
  
  ntot = 2*nc*nc;
  
  nn[0] = nn[1] = nc;
  ncD = 2 * nc;
  
  // Normalization of FFT and IFFT in such way that
  // one operator is the adjoint of the other one.
  // 1/number_elements  --->  for the FFT N-dim
  // number_elements    --->  for the IFFT N-dim
  // But, since the implementation in C of the IFFT N-dim,
  // is the matrhematical IFFT N-dim multiplied for the
  // number_elements the norm_factor is again
  // 1/number_elements as it is for the FFT N-dim
  //    Added by F. Arcadu on the 13/11/2013
  norm_factor = 1.0/(float)nc; 
  
  // Apply FFT/IFFT 2D
  fourn( data-1 , nn-1 , ndim , isign );
  
  // Normalization
  for( i=0 ; i<nc ; i++ ){
    for( j=0 ; j<ncD ; j+=2 ){
      data[i*ncD + j] *= norm_factor;
      data[i*ncD + j + 1] *= norm_factor;
    }
  }
}
