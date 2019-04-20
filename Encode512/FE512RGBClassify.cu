#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <vector>

#define  N      512
#define  N2     256
#define  Db      16
#define  Rb       8
#define  Dnum   248  //N2-Rb

using namespace cv;
using namespace std;

//Compile with opencv package
//nvcc FE512RGBClassify.cu -o FERGB512 `pkg-config --cflags --libs opencv` --expt-relaxed-constexpr
//nvprof ./FERGB512 ../Dataset/LennaColor512.tiff

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
    if(i == count){
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

__device__ int Classify(cuda::PtrStep<uchar> inputImage,int imageSize,int x,int y){
    const int BlockSize = Rb;
    int permu[BlockSize][BlockSize];
    int a1,a2,a3,a4; //Four subblock of the classify block
    int i,j,k;
    int kout;
    //Permutation
    for(k=0;k<8;k++){
        permutation(inputImage,imageSize,&permu[0][0],x,y,BlockSize,k);
        //Calculate a1,a2,a3,a4
        a1=0;
        a2=0;
        a3=0;
        a4=0;
        for(i=0;i<BlockSize;i++){
          for(j=0;j<BlockSize;j++){
              if(i < BlockSize/2 && j< BlockSize/2){
                  a1 += permu[i][j];
              }else if(i< BlockSize/2 && j >= BlockSize/2){
                  a2 += permu[i][j];
              }else if(i >= BlockSize/2 && j < BlockSize/2){
                  a3 += permu[i][j];
              }else{
                  a4 += permu[i][j];
              }
          }
        }
        //Classify by means of a1,a2,a3,a4
        if(a1>=a2 && a2>=a3 && a3>=a4){
            kout = 10+k;
        }
        if(a1>=a2 && a2>=a4 && a4 >=a3){
            kout = 20+k;
        }
        if(a1>=a4 && a4>=a2 && a2>=a3){
            kout = 30+k;
        }
    }//k
    return kout;
}

__global__ void DomainBlockClassify(cuda::PtrStep<uchar> downImage,cuda::PtrStep<uchar> Result){
    int x= blockIdx.x;
    int y= threadIdx.x; 
    if(x<Dnum && y<Dnum){
        Result(x,y) = Classify(downImage,N2,x,y);
    }
}

__device__ void calSM(int *sourceR,int* sourceD,float *destination){
    int Ud = 32;
    int m = 32;
    int i,j,ks;
    float s;
    float sup = 0.0;
    float sdown = 0.0;
    int tmpR,tmpD;
    float tmperr;
    float err = 0.005;

    //Calculate s,m,k
    for(i=0;i<Rb;i++){
        for(j=0;j<Rb;j++){
            Ud += *(sourceD+i*Rb+j);
            m += *(sourceR+i*Rb+j);
        }
    }
    Ud = Ud/(Rb*Rb);
    m = m/(Rb*Rb);
    for(i=0;i<Rb;i++){
        for(j=0;j<Rb;j++){
            tmpR = *(sourceR+i*Rb+j);
            tmpD = *(sourceD+i*Rb+j);
            sup += (tmpD-Ud)*(tmpR);
            sdown += (tmpD-Ud)*(tmpD-Ud);
        }
    }
    s= ( fabs(sdown)<0.01? 0.0 : sup/sdown);
    ks=(s<-1? 0: s>=2.1?31:(short int)(10.5+s*10));
    s=0.1*ks-1;
    for(i=0;i<Rb;i++){
        for(j=0;j<Rb;j++){
            tmpR = *(sourceR+i*Rb+j);
            tmpD = *(sourceD+i*Rb+j);
            tmperr = s*(tmpD-Ud)+ m - tmpR;
            err += (tmperr*tmperr);
        }
    }
    *(destination+0) = ks;
    *(destination+1) = m;
    *(destination+2) = err;
}

__device__ float calK(int Rk,int Dk){
    if(Rk==Dk){
        return 0;
    }else if(Rk < 4 && Dk < 4){
        if(Rk<Dk){
            if(Dk-Rk ==1){
                return 1;
            }else if(Dk-Rk ==2){
                return 2;
            }else{
                return 3;
            }
        }else{
            if(Rk-Dk==1){
                return 3;
            }else if(Rk-Dk==2){
                return 2;
            }else{
                return 1;
            }
        }
    }else if(Rk >= 4 && Dk >= 4){
        if(Rk<Dk){
            if(Dk-Rk ==1){
                return 3;
            }else if(Dk-Rk ==2){
                return 2;
            }else{
                return 1;
            }
        }else{
            if(Rk-Dk==1){
                return 1;
            }else if(Rk-Dk==2){
                return 2;
            }else{
                return 3;
            }
        }
    }else if(Rk < 4 && Dk >= 4){
        if(Dk-Rk==4){
            return 4;
        }else if(Dk-Rk==5 || Dk-Rk == 1){
            return 5;
        }else if(Dk-Rk==6 || Dk-Rk == 2){
            return 6;
        }else{
            return 7;
        }
    }else{
        if(Rk-Dk==4){
            return 4;
        }else if(Rk-Dk==5 || Rk-Dk == 1){
            return 5;
        }else if(Rk-Dk==6 || Rk-Dk == 2){
            return 6;
        }else{
            return 7;
        }
    }
}

__global__ static void RangeParallel(cuda::PtrStep<uchar> image,cuda::PtrStep<uchar> downimage,cuda::PtrStep<uchar> klass,float *Output,int Rx,int Ry){
    //int i,j,m;
    __shared__ float tmpOutput[5][Dnum];
    int tmpR[Rb][Rb];
    int tmpD[Rb][Rb];
    float destination[3];
    int Dclass,Rclass;
    int Dk,Rk;
    int mask=1;
    int offset=1;
    int x = blockIdx.x;
    int y = threadIdx.x;
    Dclass = klass(x,y)/10;
    Rclass = Classify(image,N,Rx,Ry);
    Dk = klass(x,y)%10;
    tmpOutput[4][y] = 6553500;
    if(Dclass == Rclass/10){
        Rk = Rclass%10;
        permutation(image,N,&tmpR[0][0],Rx,Ry,Rb,Rk);
        permutation(downimage,N2,&tmpD[0][0],x,y,Rb,Dk);
        calSM(&tmpR[0][0],&tmpD[0][0],&destination[0]);
        tmpOutput[0][y] = y;
        tmpOutput[1][y] = calK(Rk,Dk);
        tmpOutput[2][y] = destination[0];
        tmpOutput[3][y] = destination[1];
        tmpOutput[4][y] = destination[2];
    }
    __syncthreads();

    while(offset < Dnum){
        if((y & mask) == 0 && (y+offset) < Dnum){
            if(tmpOutput[4][y+offset] < tmpOutput[4][y]){
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
        Output[x*5]= tmpOutput[0][y];
        Output[x*5+1]= tmpOutput[1][y];
        Output[x*5+2]= tmpOutput[2][y];
        Output[x*5+3]= tmpOutput[3][y];
        Output[x*5+4]= tmpOutput[4][y];
    }
} 

int main(int argc, char** argv){
    if(!InitCUDA()) return 0;
    printf("CUDA initialized.\n");
    clock_t start, end, totaltime;
    size_t free_mem,total_mem;
    
    Mat image,downimage;
    float *output; 
    cuda::GpuMat Gpuimage,Gpudownimage;
    cuda::GpuMat Gpuclass(Dnum,Dnum,CV_8U);
    float *GpuOutput;
    cudaMalloc((void**)&GpuOutput,sizeof(float)*5*Dnum);
    output = (float*)malloc(sizeof(float)*5*Dnum);
    image = imread(argv[1],1);
    
    //Open the file for store encoding data
    fstream outfile;
    outfile.open("512RGBOutcode",ios::out);
    if(!outfile){
        cout << "Open out file fail!!" << endl;
        return 0;
    }
    
    start = clock();
       
    //Encoding
    int i,j,ll;
    int x,y,k,m,s;
    float Emin;
    Emin=6553600;
    int channel;
    Mat singleImage;
    singleImage.create(image.rows,image.cols,CV_8U);
    for(channel=0;channel<3;channel++){
        for(i=0;i<image.rows;i++){
            for(j=0;j<image.cols;j++){
                singleImage.at<uchar>(i,j) = image.at<Vec3b>(i,j)[channel];
            }
        }
        resize(singleImage,downimage,Size(singleImage.cols/2,singleImage.rows/2),0,0,INTER_LINEAR);
    
        Gpuimage.upload(singleImage);
        Gpudownimage.upload(downimage);
        //Classify the domain block into 3 class
        DomainBlockClassify<<<Dnum,Dnum>>>(Gpudownimage,Gpuclass);
    
        //For each Range, calculate s,m value
        for(i=0;i<N;i+=Rb){
            for(j=0;j<N;j+=Rb){
                RangeParallel<<<Dnum,Dnum>>>(Gpuimage,Gpudownimage,Gpuclass,GpuOutput,i,j);
                cudaMemcpy2D(output,sizeof(float)*5,GpuOutput,sizeof(float)*5,sizeof(float)*5,Dnum,cudaMemcpyDeviceToHost); 
                for(ll=0;ll<Dnum;ll++){
                    if(output[ll*5+4] <= Emin){
                        Emin = output[ll*5+4];
                        x = ll;
                        y = output[ll*5];
                        k= output[ll*5+1]; 
                        s= output[ll*5+2]; 
                        m= output[ll*5+3]; 
                    }
                }
                Emin = 6553600;
                outfile << (char)x << (char)y << (char)m << (char)((k<<5)+s);  
            }
        }
    
    }
    //Release the memory
    outfile.close();
    Gpuimage.release();
    Gpudownimage.release();
    Gpuclass.release();
    cudaFree(GpuOutput);
    free(output);
    end = clock();

    //Print time and the remain memory
    cudaError_t cudaErr;
    totaltime = end-start;
    double sec = (double) totaltime / CLOCKS_PER_SEC;
    cout <<"Time:" << sec << endl;
    cudaErr = cudaMemGetInfo(&free_mem, &total_mem);
    if(cudaErr != cudaSuccess){ 
        printf("%s in %s at line %d\n", cudaGetErrorString(cudaErr), __FILE__, __LINE__); 
    }
    cout << "free:" << free_mem << endl;
    cout << "total:" << total_mem << endl;
    return 0;
}