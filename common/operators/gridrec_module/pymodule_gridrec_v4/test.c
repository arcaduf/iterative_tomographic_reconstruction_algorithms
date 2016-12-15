#include <fftw3.h>
     int main()
     {
         int N = 4096;
         
         double *test = (double*)malloc(2*N*N*sizeof(double));
         
         fftw_complex *in, *out;
         
         fftw_plan p;
         in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N*N);
         int i,j;
         for(i=0;i<N*N;i++){
            in[i][0] = test[2*i];
            in[i][1] = test[2*i+1];
         }
         out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N*N);
         FILE *fp = fopen("wisdomM.wis","r");
         if(fp){
             fftw_import_wisdom_from_file(fp);
            fclose(fp);
        }
         p =  fftw_plan_dft_2d(N, N,
                                in, out,
                                FFTW_FORWARD, FFTW_MEASURE);
         //p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
         fftw_execute(p); /* repeat as needed */
         fp = fopen("wisdomM.wis","w");
         if(fp){
         fftw_export_wisdom_to_file(fp);
         fclose(fp);
     }
         fftw_destroy_plan(p);
         printf("%lf %lf\n",out[0][0],out[1][0]);
         fftw_free(in); fftw_free(out);
     }
