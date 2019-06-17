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

//g++ PsnrSsim.cpp -o psnrssim `pkg-config --cflags --libs opencv`
//./psnrssim 
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

double Psnr(Mat& origin,Mat& compress){
    double psnr =0;
    double mse=0;
    int tmp=0;
    if(origin.size()!=compress.size()){
        return 0;
    }
    for(int i=0;i<origin.rows;i++){
        for(int j=0;j<origin.cols;j++){
            mse += pow(origin.at<uchar>(i,j) - (int)compress.at<uchar>(i,j),2);
        }
    mse = mse/(origin.rows * origin.cols);
    }
    cout << "mse: " << mse << endl;
    tmp = pow(255,2)/mse;
    psnr = 10*log10(tmp);
    return psnr;
}

Scalar Ssim(Mat &i1, Mat & i2){
	const double C1 = 6.5025, C2 = 58.5225;
	int d = CV_32F;
	Mat I1, I2;
	i1.convertTo(I1, d);
	i2.convertTo(I2, d);
	Mat I1_2 = I1.mul(I1);
	Mat I2_2 = I2.mul(I2);
	Mat I1_I2 = I1.mul(I2);
	Mat mu1, mu2;
	GaussianBlur(I1, mu1, Size(11,11), 1.5);
	GaussianBlur(I2, mu2, Size(11,11), 1.5);
	Mat mu1_2 = mu1.mul(mu1);
	Mat mu2_2 = mu2.mul(mu2);
	Mat mu1_mu2 = mu1.mul(mu2);
	Mat sigma1_2, sigam2_2, sigam12;
	GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
	sigma1_2 -= mu1_2;
 
	GaussianBlur(I2_2, sigam2_2, Size(11, 11), 1.5);
	sigam2_2 -= mu2_2;
 
	GaussianBlur(I1_I2, sigam12, Size(11, 11), 1.5);
	sigam12 -= mu1_mu2;
	Mat t1, t2, t3;
	t1 = 2 * mu1_mu2 + C1;
	t2 = 2 * sigam12 + C2;
	t3 = t1.mul(t2);
 
	t1 = mu1_2 + mu2_2 + C1;
	t2 = sigma1_2 + sigam2_2 + C2;
	t1 = t1.mul(t2);
 
	Mat ssim_map;
	divide(t3, t1, ssim_map);
	Scalar mssim = mean(ssim_map);
 
	//double ssim = (mssim.val[0] + mssim.val[1] + mssim.val[2])/3;
	return mssim;
}

double getPSNR(const Mat& I1, const Mat& I2) { 
    Mat s1; 
    absdiff(I1, I2, s1); // |I1 - I2|AbsDiff函数是 OpenCV 中计算两个数组差的绝对值的函数 
    s1.convertTo(s1, CV_32F); // 这里我们使用的CV_32F来计算，因为8位无符号char是不能进行平方计算 
    s1 = s1.mul(s1); // |I1 - I2|^2 
    Scalar s = sum(s1); //对每一个通道进行加和 
    double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels 
    if( sse <= 1e-10) // 对于非常小的值我们将约等于0 
        return 0; 
    else 
    { 
        double mse = sse /(double)(I1.channels() * I1.total()); //计算MSE 
        double psnr = 10.0*log10((255*255)/mse); 
        return psnr;//返回PSNR 
    } 
}

int main(int argc, char** argv){
    Mat image,Encode;
	image = imread(argv[1],1);
    //Encode = imread(argv[2],1);
    double psnr = 0;
    Scalar mssim;
    double ssim = 0;
    //psnr = getPSNR(image,Encode);
    //mssim = Ssim(image,Encode);
    //ssim = (mssim.val[0] + mssim.val[1] + mssim.val[2])/3;
    cout << "psnr : " << psnr << endl;
    cout << "ssim : " << mssim << " " << ssim << endl;
    imwrite("output.jpg",image);
    return 0;
}

