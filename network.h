#ifndef NETWORK_H_INCLUDED
#define NETWORK_H_INCLUDED

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
using namespace cv;
using namespace std;


class network{
private:
    torch::Tensor result;
    torch::Tensor predraw;
    torch::Tensor image_tensor;
    torch::DeviceType device_type;
    torch::jit::script::Module module;
    std::vector<torch::jit::IValue> inputs;
    VideoCapture cap;
    Mat pic;
    Mat grey;
public:
    network(const std::string& filename);
    Mat seg_pic(Mat img);

};

#endif // NETWORK_H_INCLUDED
