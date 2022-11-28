#define USING_NETCAM 0
#include "SIGNET/network.h"
#include "CAMERA/camera_class.h"
#include <iostream>
#include <string>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/video/video.hpp>
#include <opencv2/opencv.hpp>
#include "CAMERA/algroth.cpp"

const int para_time=100;

using namespace std;
int main(int argc, char** argv)
{   Mat out_seg,out_clolored,combined,sig_feature;
    Mat cap_pic;
    Mat featurMat,old_raw,Graypre,Gray,now_raw;
    vector<Point2f>feature_pre;
    vector<Point2f>feature_after;
    vector<uchar> status;
    vector<float> err;
    cv::cuda::GpuMat gpumat1(480,640,CV_8UC3);
    double fps;
    double t0,t1;
    char fps_str[10];
    int para_count=0;
    int para_flag=0;
    string compute_mode="GPU mode";
    bool computeMd=true;
    //cout<<"argc:"<<argc<<endl;
    if(argc>1){
        if(strcmp(argv[1],"cpu")*strcmp(argv[1],"CPU")==0)
        {
            computeMd=false;
            compute_mode="CPU mode";
        }
        else
        {
            computeMd=true;
            compute_mode="GPU mode";
        }
    }
    cout<<"Predicting Mode  : "<<compute_mode<<endl;
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
#if USING_NETCAM == 0
    cap.read(cap_pic);
#else
    cam.get_pic(&unresized_cam);
    resize(unresized_cam,cap_pic,Size(640,480));
#endif

    old_raw=cap_pic.clone();
    cvtColor(cap_pic,Gray,COLOR_BGR2GRAY);
    Graypre=Gray.clone();
    while(1)
    {
        t0=double(getTickCount());
#if USING_NETCAM == 0
        cap.read(cap_pic);
#else
        cam.get_pic(&unresized_cam);
        resize(unresized_cam,cap_pic,Size(640,480));
#endif

        cap_pic=Sharpen(cap_pic,40,0);
        //imshow("input_cam",cap_pic);
        out_seg=seg_net.seg_pic(cap_pic);
        sig_feature=out_seg.clone();
        applyColorMap(sig_feature*10,out_clolored,COLORMAP_JET);
        addWeighted(cap_pic,0.7,out_clolored,0.3,0,combined);


        now_raw=cap_pic.clone();
        cvtColor(cap_pic,Gray,COLOR_BGR2GRAY);
        catchfeature(Gray, featurMat, feature_pre,256);
        calcOpticalFlowPyrLK(Graypre, Gray, feature_pre, feature_after, status, err);

        int k = 0;
        for (int i = 0; i < feature_after.size(); i++)
        {
            //状态要是1，并且坐标要移动下的那些点
            if (status[i] && ((abs(feature_pre[i].x - feature_after[i].x) +
                               abs(feature_pre[i].y - feature_after[i].y)) > 1))
            {
                feature_after[k++] = feature_after[i];
            }
        }
        feature_after.resize(k);//截取
        //cout << k << endl;
        for (int i = 0; i < feature_after.size(); i++)
        {
            //将特征点画一个小圆出来--粗细为2
            imshow("sigcopy",sig_feature);
            if((int)sig_feature.at<uchar>(feature_after[i])!=12 && (int)sig_feature.at<uchar>(feature_after[i])!=13 && (int)sig_feature.at<uchar>(feature_after[i])<13)
            {
                circle(now_raw, feature_after[i], 3, Scalar(0,255,255), 2);
            }

        }
        imshow("oldraw",now_raw);





        t1=double(getTickCount());
        fps=1.0/((t1-t0)/getTickFrequency());
        sprintf(fps_str,"%0.2f",fps);
        string fpsString("FPS:");
        fpsString+=fps_str;
        putText(combined,fpsString,Point(5,20),FONT_HERSHEY_COMPLEX,0.5,Scalar(0,255,0));
        imshow("result",out_clolored);
        imshow("combined",combined);
        old_raw=cap_pic.clone();
        Graypre=Gray.clone();
#if USING_NETCAM == 1

        if(para_flag==1){
            cam.re_iso();
            para_count++;
            usleep(5000);
            if(para_count>=para_time)
            {
                para_flag=0;
                para_count=0;
            }
        }
#endif
        int key=waitKey(1);
        if (key==27)
        {
#if USING_NETCAM == 1
            cam.close_cam();
#endif
            sleep(1);
            break;
        }else if(key==116){
            para_flag=1;
        }

    }
}


