#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

using namespace std;

//nvcc cudaProperty.cu -o cudaProp


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

int main(int argc, char** argv){
    cudaDeviceProp prop;
        if(cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
            if(prop.major >= 1) {
                cout << "Name : " << prop.name << endl;
                cout << "Total Global Mem : " << prop.totalGlobalMem << endl;
                cout << "Shared Mem per block : " << prop.sharedMemPerBlock<< endl; 
                cout << "Max Thread per block : " << prop.maxThreadsPerBlock<< endl; 
                cout << "total Const Mem : " << prop.totalConstMem<< endl; 
                cout << "multiProcessor : " << prop.multiProcessorCount<< endl; 
                cout << "Warp Size : " << prop.warpSize<< endl; 
                cout << "ClockRate : " << prop.clockRate<< endl;    
                cout << "Major : " << prop.major<< endl; 
                cout << "Minor : " << prop.minor<< endl; 
            }
        }
    return 0;
}