#include "network.h"
network::network(const std::string& filename,bool using_gpu)
{
    device_type = at::kCPU; // 定义设备类型
    if(using_gpu)
    {
        std::cout <<"cuda::is_available: " << torch::cuda::is_available() << std::endl;
        std::cout <<"cudnn::is_available: " << torch::cuda::cudnn_is_available() << std::endl;
        // Deserialize the ScriptModule from a file using torch::jit::load().
        if (torch::cuda::is_available())device_type = at::kCUDA;
    }
    cout<<"using device: "<<device_type<<endl;
    module = torch::jit::load(filename);
    module.eval();
    module.to(device_type);
    cout<<"finish loading model"<<endl;
}
cv::Mat network::seg_pic(cv::Mat img)
{
    Mat grey;
    cvtColor(img,grey,COLOR_BGR2RGB);
    image_tensor= torch::from_blob(grey.data,{1,grey.rows,grey.cols,3},torch::kByte);
    image_tensor=image_tensor.permute({0,3,1,2});
    image_tensor=image_tensor.toType(torch::kFloat);
    image_tensor=image_tensor.div(255);
    image_tensor[0][0]=image_tensor[0][0].sub_(0.485).div_(0.229);
    image_tensor[0][1]=image_tensor[0][1].sub_(0.456).div_(0.224);
    image_tensor[0][2]=image_tensor[0][2].sub_(0.406).div_(0.225);
    image_tensor=image_tensor.to(device_type);
    //cout<<"image is leaf:"<<image_tensor.is_leaf()<<endl;

    inputs.emplace_back(image_tensor);
    torch::NoGradGuard no_grad;
    result=module.forward(inputs).toTuple()->elements()[0].toTensor();
    predraw=result.argmax(1);
    torch::Tensor pred=predraw.squeeze();
    pred=pred.to(torch::kU8);
    pred=pred.to(torch::kCPU);
    pred=pred*10;
    inputs.pop_back();
    Mat outimg(Size{640,480},CV_8U,pred.data_ptr());
    return outimg;
}
