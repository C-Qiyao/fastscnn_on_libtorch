#include "network.h"
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/video/video.hpp>
#include <opencv2/opencv.hpp>
using namespace std;
int main()
{
    VideoCapture cap;
    cap.open(0);
    Mat cap_pic;
    Mat out_seg;
    cout<<"starting test"<<endl;
    network seg_net("/home/qiyao/python_codes/fastscnn_cv/model_scnn.pt");
    while(1)
    {
        cap.read(cap_pic);
        imshow("test",cap_pic);
        out_seg=seg_net.seg_pic(cap_pic);
        imshow("result",out_seg);
        int k=waitKey(1);
        if (k==27)
        {
    //cout <<pred;
        break;
        }

    }
}


