# fastscnn_on_libtorch
fastscnn saved by pytorch and deployed on C++ with libtorch

本项目是一个将pytorch部署至C++的尝试 

# 软件环境版本
UBUNTU=22.04

pytorch=1.13.0

libtorch=1.13.0

CUDA=11.7.99

cudnn=8.5.0.96
# 文件说明
main.cpp 为主函数

py文件为转换模型所使用的实例，将pytorch内的网络转换成.pt文件

./SIGNET 文件夹内包含了神经网络网络类的操作函数

./CAMERA 文件夹内包含了海康工业相机的操作函数

./MVS  文件夹内包含了海康工业相机的驱动库

# 测试效果（模型尚未训练完备）

![net_test](https://user-images.githubusercontent.com/74750146/203498742-a1c6e2b7-050a-4bb3-99ff-148f90f1b881.png)
![截图 2022-11-24 13-01-38](https://user-images.githubusercontent.com/74750146/203706585-32c040e9-cb8e-4930-993c-ee3aa714a622.png)

# Demo on Game  Forza horizon 5

https://user-images.githubusercontent.com/74750146/204536524-b7365870-9c70-47fc-a3ae-37cd832a0d57.mp4


## *使用之前需要修改Cmakelist.txt 内的工程目录地址，以及main函数内加载神经网络的文件路径

[![Security Status](https://s.murphysec.com/badge/C-Qiyao/fastscnn_on_libtorch.svg)](https://www.murphysec.com/p/C-Qiyao/fastscnn_on_libtorch)

