#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define  N   1024
#define  N2   512
#define  Db    16
#define  Rb     8
#define  nRun  20


using namespace cv;
using namespace std;

// g++ Decoding1024.cpp -o FD1024 `pkg-config --cflags --libs opencv`
// ./FD1024 ../1024Outcode 

typedef struct code{
    int x;
    int y;
    int k;
    short int ns;
    int m;
}code;

void permutation(const int *h,int *a,int R,int k){
    //x,y is the position in the Mat h.
    //R is the size of array a.
    int i1,j1;
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
                  *(a+i1*R+j1)= *(h+(j1)*R+i1);
              break;

           } /* end switch */
}


void Decode(vector<code> *inputcode,Mat &DecodeImage){
    Mat down;
    down.create(N2,N2,CV_8U);
    int i,j,ii,jj,nn;
    int x,y,k,u,Dmean;
    float s;
    int D[Rb][Rb],PD[Rb][Rb];
    int tmpoutput[N][N];

    for(i=0;i<N;i++){
        for(j=0;j<N;j++){
            DecodeImage.at<uchar>(i,j) = 50;
        }
    }
    
    for(int n=0;n < nRun;n++){
        //Downsample the image
        resize(DecodeImage,down,Size(DecodeImage.cols/2,DecodeImage.rows/2),0,0,INTER_LINEAR);
        //For each block decode
        nn=0;
        for(i=0;i<N;i+=Rb){
            for(j=0;j<N;j+=Rb){
                Dmean=0;
                x= inputcode->at(nn).x;
                y= inputcode->at(nn).y;
                k= inputcode->at(nn).k;
                u= inputcode->at(nn).m;
                s= 0.10*(inputcode->at(nn).ns)-1.0; 
                for(ii=0;ii<Rb;ii++){
                    for(jj=0;jj<Rb;jj++){
                        D[ii][jj] = down.at<uchar>(x+ii,y+jj);
                        Dmean += D[ii][jj]; 
                    }
                }
                Dmean = Dmean/(Rb*Rb);
                permutation(&D[0][0],&PD[0][0],Rb,k);
                for(ii=0;ii<Rb;ii++){
                    for(jj=0;jj<Rb;jj++){
                        tmpoutput[i+ii][j+jj] = s * (PD[ii][jj]-Dmean) + u;
                    }
                }
                nn++;
            }   
        }
        //Copy to the decode image
        for(i=0;i<N;i++){
            for(j=0;j<N;j++){
                tmpoutput[i][j]=(tmpoutput[i][j] > 255 ? 255 : tmpoutput[i][j]< 0 ? 0 : tmpoutput[i][j]); 
                DecodeImage.at<uchar>(i,j)=tmpoutput[i][j];
            }
        }
    } //nRun
}

int main(int argc, char** argv){
    char x,y,m,ns;
    short int byte2;
    int i,j;
    code c;
    vector<code> input;
    //Read the code file
    fstream infile;
    infile.open(argv[1],ios::in);
    if(!infile){
        cout << "Input file fail!" << endl;
        return 1;
    }
    while(infile){
            infile.get(x);
            infile.get(y);
            infile.get(m);
            infile.get(ns);
            c.x = (int)(unsigned char)x;
            c.y = (int)(unsigned char)y;
            c.m = (int)(unsigned char)m;
            byte2 = (short int)(unsigned char)ns;
            c.k = (byte2>>5);
            c.ns = (byte2&31);  
            input.push_back(c);
    }
    infile.close();

    //Decode the image
    Mat Dimage;
    Dimage.create(N,N,CV_8U);
    Decode(&input,Dimage);
    imshow("Display",Dimage);
    imwrite("512FEImage.tiff",Dimage);
    waitKey(0);
}

