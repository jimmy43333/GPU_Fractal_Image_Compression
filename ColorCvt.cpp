#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <vector>

//  g++ ColorCvt.cpp -o ColorCvt `pkg-config --cflags --libs opencv`

using namespace cv;
using namespace std;

int main(int argc, char** argv){

    Mat image,oimage,downU,downV,Dimage;
    vector<Mat> rgbchannels(3);
    vector<Mat> tmp3channel(3); 
    image = imread(argv[1],1);
    cvtColor(image,oimage,CV_BGR2YCrCb);
    split(oimage,rgbchannels);
    split(image,tmp3channel);
    resize(rgbchannels[1],downU,Size(image.cols/2,image.rows/2),0,0,INTER_LINEAR);
    resize(rgbchannels[2],downV,Size(image.cols/2,image.rows/2),0,0,INTER_LINEAR);
    //imshow("Display",rgbchannels[1]);
    //imshow("Down",downU);
    
    

    for(int i=0;i<image.rows;i++){
        for(int j=0;j<image.cols;j++){
            tmp3channel[0].at<uchar>(i,j) = rgbchannels[0].at<uchar>(i,j);
            tmp3channel[1].at<uchar>(i,j) = downU.at<uchar>(i/2,j/2);
            tmp3channel[2].at<uchar>(i,j) = downV.at<uchar>(i/2,j/2);
        }
    }
    imshow("Origin",rgbchannels[1]);
    imshow("After",tmp3channel[1]);
    merge(tmp3channel,Dimage);
    cvtColor(Dimage,oimage,CV_YCrCb2BGR);
    imshow("Dimage",oimage);
    waitKey(0);
    
    return 0;
}