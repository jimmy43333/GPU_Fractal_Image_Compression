#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>

#define  N    512
#define  N2   256
#define  Db    16
#define  Rb     8
#define  Dnum 248 
#define  nRun  10


using namespace cv;
using namespace std;

//g++ psnr.cpp -o psnr `pkg-config --cflags --libs opencv`
//./FracDecoding512 ./Baseline_FIC/baselineOutcode

double psnr(Mat& origin,Mat& compress){
    double psnr =0;
    double mse=0;
    int tmp=0;
    if(origin.size()!=compress.size()){
        return 0;
    }
    for(int i=0;i<origin.rows;i++){
        for(int j=0;j<origin.cols;j++){
            tmp = origin.at<uchar>(i,j) - compress.at<uchar>(i,j);
            mse += (tmp*tmp);
        }
    }
    mse = mse/(origin.rows * origin.cols);
    psnr = 10*log10((255*255)/mse);
    return psnr;
}

int main(int argc, char** argv){
    Mat image,Encode;
    image = imread(argv[1],0);
    Encode = imread(argv[2],0);
    double p = 0;
    p = psnr(image,Encode);
    cout << p << endl;
    return 0;
}

