#include "torch/torch.h"
#include <torch/script.h> // One-stop header.
#include <tuple>
#include <iostream>
#include <memory>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/video/video.hpp>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
int main(int argc, const char *argv[])
{
    Mat pic;
    Mat grey;
    VideoCapture cap;
    cap.open(0);
    torch::Tensor result;
    torch::Tensor predraw;
    torch::Tensor image_tensor;




    std::cout <<"cuda::is_available():" << torch::cuda::is_available() << std::endl;
    // Deserialize the ScriptModule from a file using torch::jit::load().
    torch::DeviceType device_type = at::kCPU; // 定义设备类型
    if (torch::cuda::is_available())device_type = at::kCUDA;
    torch::jit::script::Module module = torch::jit::load("/home/qiyao/python_codes/fastscnn_cv/model_scnn.pt",device_type);
    module.eval();
    module.to(device_type);


    //for(int i=0;i<4;i++){
    while(1){
    cap.read(pic);
    imshow("test",pic);


    cvtColor(pic,grey,COLOR_BGR2RGB);
    image_tensor= torch::from_blob(grey.data,{1,grey.rows,grey.cols,3},torch::kByte);
    image_tensor=image_tensor.permute({0,3,1,2});
    image_tensor=image_tensor.toType(torch::kFloat);
    image_tensor=image_tensor.div(255);
    image_tensor=image_tensor.to(device_type);
    //cout<<"image is leaf:"<<image_tensor.is_leaf()<<endl;

    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(image_tensor);
    torch::NoGradGuard no_grad;
    result=module.forward(inputs).toTuple()->elements()[0].toTensor();
    predraw=result.argmax(1);
    torch::Tensor pred=predraw.squeeze();
    pred=pred.to(torch::kU8);
    pred=pred.to(torch::kCPU);
    pred=pred*10;
    //inputs.pop_back();
    Mat outimg(Size{640,480},CV_8U,pred.data_ptr());

    imshow("label",outimg);

    int k=waitKey(1);
    if (k==27)
    {
    //cout <<pred;
    break;
    }
}
    std::cout<< "ok\n";
}


