#include "Layers.hpp"
#include <vector>

InvertedResidualImpl::InvertedResidualImpl(const int c_in, const int c_out, const int s, const double expand_ratio): EXPAND_RATIO(expand_ratio){
    if(s==1 && c_in==c_out)
        identity = true;
    else
        identity = false;

    if(expand_ratio == 1){
        conv->push_back(register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(c_in, c_in, 3).stride(s).padding(1).groups(c_in).bias(false))));
        conv->push_back(register_module("bn1", torch::nn::BatchNorm2d(c_in)));
        conv->push_back(register_module("act1", torch::nn::ReLU6(torch::nn::ReLU6Options().inplace(true))));

        conv->push_back(register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(c_in, c_out, 1).stride(1).padding(0).bias(false))));
        conv->push_back(register_module("bn2", torch::nn::BatchNorm2d(c_out)));
    }
    else{
        conv->push_back(register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(c_in, c_in*expand_ratio, 1).stride(1).padding(0).bias(false))));
        conv->push_back(register_module("bn1", torch::nn::BatchNorm2d(c_in*expand_ratio)));
        conv->push_back(register_module("act1", torch::nn::ReLU6(torch::nn::ReLU6Options().inplace(true))));

        conv->push_back(register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(c_in*expand_ratio, c_in*expand_ratio, 3).stride(s).padding(1).groups(c_in*expand_ratio).bias(false))));
        conv->push_back(register_module("bn2", torch::nn::BatchNorm2d(c_in*expand_ratio)));
        conv->push_back(register_module("act2", torch::nn::ReLU6(torch::nn::ReLU6Options().inplace(true))));

        conv->push_back(register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(c_in*expand_ratio, c_out, 1).stride(1).padding(0).bias(false))));
        conv->push_back(register_module("bn3", torch::nn::BatchNorm2d(c_out)));
    }
}

torch::Tensor InvertedResidualImpl::forward(torch::Tensor x){
    if(identity)
        return conv->forward(x) + x;
    return conv->forward(x);
}

ConvImpl::ConvImpl(const int c_in, const int c_out, const int k, const int s, const int p){
    conv = register_module("conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(c_in, c_out, k).stride(s).padding(p).bias(false)));
    bn = register_module("bn", torch::nn::BatchNorm2d(c_out));
    act = register_module("act", torch::nn::ReLU6(torch::nn::ReLU6Options()));
}

torch::Tensor ConvImpl::forward(torch::Tensor x){
    x = conv->forward(x);
    x = bn->forward(x);
    return act->forward(x);
}

BackboneImpl::BackboneImpl(){
    int c_in = 32;
    int c_out;
    conv1 = register_module("conv1", Conv(3, c_in, 3, 2, 1));
    for(int i=0; i<7; i++){
        int t = BLOCKS[i][0];
        int c = BLOCKS[i][1];
        int n = BLOCKS[i][2];
        int s = BLOCKS[i][3];
        c_out = c;
        for(int j=0; j<n; j++){
            res.push_back(register_module("res"+std::to_string(i)+std::to_string(j), InvertedResidual(c_in, c_out, s, t)));
            c_in = c_out;
            s = 1;
        }
    }
    c_out = 1280;
    conv2 = register_module("conv2", Conv(c_in, c_out, 1, 1, 0));
}

torch::Tensor BackboneImpl::forward(torch::Tensor x){
    x = conv1->forward(x);
    for(int i=0; i<res.size(); i++)
        x = res[i]->forward(x);
    return conv2->forward(x);
}


HeadImpl::HeadImpl(const int num_boxes, const int num_classes, const int grid_size): NUM_BOXES(num_boxes), NUM_CLASSES(num_classes), GRID_SIZE(grid_size){
    detect1 = register_module("detect1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1280, 1280, 3).stride(1).padding(1).bias(false).groups(1280)));
    bn1 = register_module("bn1", torch::nn::BatchNorm2d(1280));
    act1 = register_module("act1", torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1).inplace(true)));
    detect2 = register_module("detect2", torch::nn::Conv2d(torch::nn::Conv2dOptions(1280, 1280, 1).stride(1).padding(0).bias(false)));
    bn2 = register_module("bn2", torch::nn::BatchNorm2d(1280));
    act2 = register_module("act2", torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1).inplace(true)));
    detect3 = register_module("detect3", torch::nn::Conv2d(torch::nn::Conv2dOptions(1280, 1024, 1).stride(1).padding(0).bias(false)));
    bn3 = register_module("bn3", torch::nn::BatchNorm2d(1024));
    act3 = register_module("act3", torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1).inplace(true)));
    detect4 = register_module("detect4", torch::nn::Conv2d(torch::nn::Conv2dOptions(1024, NUM_BOXES*5+NUM_CLASSES, 1).stride(1).padding(0)));
}

torch::Tensor HeadImpl::forward(torch::Tensor x){
    x = detect1->forward(x);
    x = bn1->forward(x);
    x = act1->forward(x);
    x = detect2->forward(x);
    x = bn2->forward(x);
    x = act2->forward(x);
    x = detect3->forward(x);
    x = bn3->forward(x);
    x = act3->forward(x);
    x = detect4->forward(x);
    return x.permute({0, 2, 3, 1});
}