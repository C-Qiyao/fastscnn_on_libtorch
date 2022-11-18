#define USING_NETCAM 1
#include "network.h"
#include "camera_class.h"
#include <iostream>
#include <string>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/video/video.hpp>
#include <opencv2/opencv.hpp>


using namespace std;
int main(int argc, char** argv)
{   Mat out_seg,out_clolored,combined;
    Mat cap_pic;
    double fps;
    double t0,t1;
    char fps_str[10];

    string compute_mode;
    bool computeMd=false;
    if(argc>=1){

        if(strcmp(argv[1],"gpu")*strcmp(argv[1],"GPU")==0)
        {
            computeMd=true;
            compute_mode="GPU mode";
        }
        else
        {
            computeMd=false;
            compute_mode="CPU mode";
        }
    }
    cout<<"Predicting Mode : "<<compute_mode<<endl;

#if USING_NETCAM == 0
    VideoCapture cap;
    cap.open(0);
#else
    camera cam;
    cam.start_cam();
    cam.PrintDeviceInfo();
    cam.re_iso();
    Mat unresized_cam;
#endif
    cout<<"starting test"<<endl;
    network seg_net("/home/qiyao/python_codes/fastscnn_cv/model_scnn.pt",computeMd);//false=using cpu //true=using gpu(if gpu is available)
    while(1)
    {
        t0=double(getTickCount());
#if USING_NETCAM == 0
        cap.read(cap_pic);
#else
        cam.get_pic(&unresized_cam);
        resize(unresized_cam,cap_pic,Size(640,480));
#endif
        imshow("input_cam",cap_pic);
        out_seg=seg_net.seg_pic(cap_pic);
        applyColorMap(out_seg,out_clolored,COLORMAP_JET);
        addWeighted(cap_pic,0.7,out_clolored,0.3,0,combined);
        t1=double(getTickCount());
        fps=1.0/((t1-t0)/getTickFrequency());
        sprintf(fps_str,"%0.2f",fps);
        string fpsString("FPS:");
        fpsString+=fps_str;
        putText(combined,fpsString,Point(5,20),FONT_HERSHEY_COMPLEX,0.5,Scalar(0,255,0));
        imshow("result",out_clolored);
        imshow("combined",combined);
        int k=waitKey(1);
        if (k==27)
        {
            //cout<<out_clolored.cols<<out_clolored.rows<<endl;
            //cout<<cap_pic.cols<<cap_pic.rows<<endl;
#if USING_NETCAM == 1
            cam.close_cam();
#endif
            sleep(1);
            break;
        }

    }
}


