#include<iostream>
#include <stdio.h>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

void catchfeature(Mat input, Mat &output,vector<Point2f> &feature,int thresh)
{
    Mat dst = Mat::zeros(input.size(), CV_32FC1);
    Mat raw = input.clone();
    vector<Point2f>corners;
    goodFeaturesToTrack(input, corners, thresh, 0.01, 10, Mat());
    for (int i = 0; i < corners.size(); i++)
    {
        circle(raw, corners[i], 2, Scalar(255, 0, 0), 2);
    }
    output = raw;
    feature = corners;
}

Mat Sharpen(Mat input, int percent, int type)
{
    Mat result;
    Mat s = input.clone();
    Mat kernel;
    switch (type)
    {
    case 0:
        kernel = (Mat_<int>(3, 3) <<
            0, -1, 0,
            -1, 4, -1,
            0, -1, 0
            );
    case 1:
        kernel = (Mat_<int>(3, 3) <<
            -1, -1, -1,
            -1, 8, -1,
            -1, -1, -1
            );
    default:
        kernel = (Mat_<int>(3, 3) <<
            0, -1, 0,
            -1, 4, -1,
            0, -1, 0
            );
    }
    filter2D(s, s, s.depth(), kernel);
    result = input + s * 0.01 * percent;
    return result;
}
