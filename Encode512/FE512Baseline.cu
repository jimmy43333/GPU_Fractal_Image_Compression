#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <vector>

#define  N       512
#define  N2      256
#define  Db       16
#define  Rb        8
#define  Dnum    248  //N2-Rb


using namespace cv;
using namespace std;

//Run on terminal:
//  nvcc FE512Baseline.cu -o FE512 `pkg-config --cflags --libs opencv`
//  nvprof ./FE512 ../Dataset/LennaGray512.tif

Mat readRawfile(const char* filename,int width,int height){
    Mat outputimage;
    //read the raw file
    FILE *fp = NULL;
    char *imagedata = NULL;
    int IMAGE_WIDTH = width;
    int IMAGE_HEIGHT = height;
    int framesize = IMAGE_WIDTH * IMAGE_HEIGHT;
    //Open raw Bayer image.
    fp = fopen(filename, "rb");
    if(!fp){
        cout << "read file failure";
        return outputimage;
    }
    //Memory allocation for bayer image data buffer.
    imagedata = (char*) malloc (sizeof(char) * framesize);
    //Read image data and store in buffer.
    fread(imagedata, sizeof(char), framesize, fp);
    //Create Opencv mat structure for image dimension. For 8 bit bayer, type should be CV_8UC1.
    outputimage.create(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1);
    memcpy(outputimage.data, imagedata, framesize);
    free(imagedata);
    fclose(fp);
    return outputimage;
}

bool InitCUDA()
{
    int count;
    cudaGetDeviceCount(&count);
    if(count == 0) {
        cout << "There is no device."<< endl;
        return false;
    }
    int i;
    for(i = 0; i < count; i++) {
        cudaDeviceProp prop;
        if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            if(prop.major >= 1) {
                break;
            }
        }
    }
    if(i == count) {
        cout << "There is no device supporting CUDA 1.x." << endl;
        return false;
    }
    cudaSetDevice(i);
    return true;
}

__device__ void permutation(cuda::PtrStep<uchar> input,int inputSize,int *a,int x,int y,int R,int k){
    //x,y is the position in the Mat h.
    //R is the size of array a.
    if(x+R <= inputSize && y+R <= inputSize){
        int i1,j1;
        int h[Rb*Rb];
        for (i1=0; i1<R; i1++){
            for (j1=0; j1<R; j1++){
                *(h+i1*R+j1) = input(x+i1,y+j1);
            }
        }
        switch(k){
               case 0: 
                 for (i1=0; i1<R; i1++)
                   for (j1=0; j1<R; j1++)
                     *(a+i1*R+j1) = *(h+i1*R+j1);
                 break;
               case 1:
                 for (i1=0; i1<R; i1++)
                  for (j1=0; j1<R; j1++)
                   *(a+i1*R+j1)= *(h+(R-1-j1)*R+i1);
                break;
             case 2:
                 for (i1=0; i1<R; i1++)
                   for (j1=0; j1<R; j1++)
                     *(a+i1*R+j1)= *(h+(R-1-i1)*R+R-1-j1);
                 break;
            case 3:
                 for (i1=0; i1<R; i1++)
                   for (j1=0; j1<R; j1++)
                    *(a+i1*R+j1)= *(h+j1*R+R-1-i1);
                 break;
                                     /* Reflect w.r.t. y-axis,  then rotate 
                                        counterclockwise 90, 180, 270 degree(s)
                                     */
            case 4:
                 for (i1=0; i1<R; i1++)
                   for (j1=0; j1<R; j1++)
                     *(a+i1*R+j1)= *(h+i1*R+R-1-j1);
                 break;
            case 5:
                  for (i1=0; i1<R; i1++)
                    for (j1=0; j1<R; j1++)
                      *(a+i1*R+j1)= *(h+(R-1-j1)*R+R-1-i1);
                 break;
            case 6:
                 for (i1=0; i1<R; i1++)
                   for (j1=0; j1<R; j1++)
                     *(a+i1*R+j1)= *(h+(R-1-i1)*R+j1);
                 break;
            case 7:
              for (i1=0; i1<R; i1++)
                for (j1=0; j1<R; j1++)
                  *(a+i1*R+j1)= *(h+j1*R+i1);
              break;

        } /* end switch */
    }
}

__global__ static void CalSM(cuda::PtrStep<uchar> Original, cuda::PtrStep<uchar> Downsample,float* Output,int Ri,int Rj,int RangeSize){
    __shared__ float tmpOutput[5][Dnum];
    int tmp[Rb][Rb];
    int i,j,k,Ud,m;
    short int ks;
    float s,sup,sdown;
    float err,tmperr,minerr;
    int offset=1;
    int mask=1;
    const int x = blockIdx.x;
    const int y = threadIdx.x;
    minerr=6553600;
    
    if(x<Dnum && y<Dnum){
        Ud=32;
        m=32;
        for(i=0;i < RangeSize;i++){
            for(j=0;j < RangeSize;j++){
                Ud += Downsample(x+i,y+j);
                m += Original(Ri+i,Rj+j);
            }
        }
        Ud = Ud/(RangeSize*RangeSize);
        m = m/(RangeSize*RangeSize);
        for(k=0;k<8;k++){
            permutation(Downsample,N2,&tmp[0][0],x,y,Rb,k);
            sup=0;
            sdown=0;
            //Calculate s,m,k
            for(i=0;i<RangeSize;i++){
                for(j=0;j<RangeSize;j++){
                    sup += (tmp[i][j]-Ud)*Original(Ri+i,Rj+j);
                    sdown += (tmp[i][j]-Ud)*(tmp[i][j]-Ud);
                }
            }
            s=(fabs(sdown)<0.01? 0.0 : sup/sdown);
            ks=(s<-1? 0: s>=2.1? 31:(short int)(10.5+s*10));
            s=0.1*ks-1;
            err=0.005;
            for(i=0;i<RangeSize;i++){
                for(j=0;j<RangeSize;j++){
                    tmperr = s*(tmp[i][j]-Ud)+ m -Original(Ri+i,Rj+j);
                    err += tmperr*tmperr;
                    if (err >= minerr){ 
                        break;
                    }
                }
                if (err >= minerr){ 
                    break;
                }
            }
            if(err < minerr){
                minerr = err;
                tmpOutput[0][y]=k;
                tmpOutput[1][y]=ks;
            }    
        }
        
        tmpOutput[2][y]=m;
        tmpOutput[3][y]=minerr;
        tmpOutput[4][y]=y;
        __syncthreads();

        while(offset < Dnum){
            if((y & mask) == 0 && (y+offset) < Dnum){
                if(tmpOutput[3][y+offset] < tmpOutput[3][y]){
                    tmpOutput[0][y] = tmpOutput[0][y+offset];
                    tmpOutput[1][y] = tmpOutput[1][y+offset];
                    tmpOutput[2][y] = tmpOutput[2][y+offset];
                    tmpOutput[3][y] = tmpOutput[3][y+offset];
                    tmpOutput[4][y] = tmpOutput[4][y+offset];   
                }
                
            }
            offset += offset;
            mask = offset + mask;
            __syncthreads();
        }
        
        if(y==0){
            Output[x*5]= tmpOutput[4][y];
            Output[x*5+1]= tmpOutput[0][y];
            Output[x*5+2]= tmpOutput[1][y];
            Output[x*5+3]= tmpOutput[2][y];
            Output[x*5+4]= tmpOutput[3][y];
        }
    }   
}

int main(int argc, char** argv){
    if(!InitCUDA()) return 0;
    fstream outfile;
    clock_t start, end;
    Mat image,downsample;
    cuda::GpuMat gpuImage,gpuDownsample;
    float *gpuOutput;
    float *output; 
    int i,j,ll;
    float Emin;
    Emin=6553600;
    int x,y,tau,ns,u;

    //Input image and DownSampling the inputdata
    image = imread(argv[1],0);
    //image.create(N,N,CV_8U);
    //image = readRawfile(argv[1],N,N);
    downsample.create(N2,N2,CV_8U);
    resize(image,downsample,Size(image.cols/2,image.rows/2),0,0,INTER_LINEAR);
    start = clock();
    
    //Pass the data to GPU
    output = (float*)malloc(sizeof(float)*5*Dnum);
    cudaMalloc((void**)&gpuOutput,sizeof(float)*5*Dnum);
    gpuImage.upload(image);
    gpuDownsample.upload(downsample);
    //Open the file for store encoding data
    outfile.open("512Outcode",ios::out);
    if(!outfile){
        cout << "Open out file fail!!" << endl;
        return 0;
    }

    //Start : For Loop of each Range block
    for(i=0;i<N;i+=Rb){
        for(j=0;j<N;j+=Rb){
            CalSM<<<Dnum,Dnum>>>(gpuImage,gpuDownsample,gpuOutput,i,j,Rb);
            cudaMemcpy2D(output,sizeof(float)*5,gpuOutput,sizeof(float)*5,sizeof(float)*5,Dnum,cudaMemcpyDeviceToHost); 
            for(ll=0;ll<Dnum;ll++){
                if(output[ll*5+4] < Emin){
                    Emin = output[ll*5+4];
                    x = ll;
                    y = output[ll*5];
                    tau= output[ll*5+1]; 
                    ns= output[ll*5+2]; 
                    u= output[ll*5+3]; 
                }
            }
            Emin = 6553600;
            outfile << (char)x << (char)y << (char)u << (char)((tau<<5)+ns);        
        }
    }
    outfile.close();
    gpuImage.release();
    gpuDownsample.release();
    cudaFree(gpuOutput);
    free(output);

    end = clock();

    //Print time and the remain memory
    clock_t time = end-start;
    double sec = (double) time / CLOCKS_PER_SEC;
    cout <<"Time:" << sec << endl;
    size_t free_mem,total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    cout << "free:" << free_mem << endl;
    cout << "total:" << total_mem << endl;
    return 0;
}