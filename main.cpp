#include <iostream>
#include "torch/script.h"
#include "torch/torch.h"
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>
using namespace std;
using namespace torch;

int main() {
    cout << torch::cuda::is_available() << endl;
    cout << torch::cuda::cudnn_is_available() << endl;
    cout << torch::cuda::device_count() << endl;
    Tensor x=tensor(1.0,requires_grad(true));
    Tensor y=tensor(2.0,requires_grad(true));
    Tensor z=x*x+y;
    z.backward();
    cout<<x.grad();



}
