#ifndef LAYERS_HPP
#define LAYERS_HPP

#include <torch/torch.h>
#include <vector>
#include <tuple>

class ConvImpl: public torch::nn::Module{
    public:
        ConvImpl(const int c_in, const int c_out, const int k, const int s, const int p);

        torch::Tensor forward(torch::Tensor x);
    private:
        torch::nn::Conv2d conv{nullptr};
        torch::nn::BatchNorm2d bn{nullptr};
        torch::nn::ReLU6 act{nullptr};
};
TORCH_MODULE(Conv);

class InvertedResidualImpl: public torch::nn::Module{
    public:
        InvertedResidualImpl(const int c_in, const int c_out, const int s, const double expand_ratio);

        torch::Tensor forward(torch::Tensor x);

    private:
        bool identity;
        const double EXPAND_RATIO;
        torch::nn::Sequential conv;
};
TORCH_MODULE(InvertedResidual);

// Mobilenet architecture
class BackboneImpl: public torch::nn::Module{
    public:
        BackboneImpl();

        torch::Tensor forward(torch::Tensor x);

    private:
        Conv conv1{nullptr};
        Conv conv2{nullptr};
        std::vector<InvertedResidual> res;
        const int BLOCKS[7][4] = {{1, 16, 1, 1}, {6, 24, 2, 2}, {6, 32, 3, 2}, {6, 64, 4, 2}, {6, 96, 3, 1}, {6, 160, 3, 2}, {6, 320, 1, 1}};
};
TORCH_MODULE(Backbone);

class HeadImpl: public torch::nn::Module{
    public:
        HeadImpl(const int num_boxes, const int num_classes, const int grid_size);

        torch::Tensor forward(torch::Tensor x);
    private:
        const int NUM_BOXES, NUM_CLASSES, GRID_SIZE;
        torch::nn::Conv2d detect1{nullptr};
        torch::nn::BatchNorm2d bn1{nullptr};
        torch::nn::LeakyReLU act1{nullptr};
        torch::nn::Conv2d detect2{nullptr};
        torch::nn::BatchNorm2d bn2{nullptr};
        torch::nn::LeakyReLU act2{nullptr};
        torch::nn::Conv2d detect3{nullptr};
        torch::nn::BatchNorm2d bn3{nullptr};
        torch::nn::LeakyReLU act3{nullptr};
        torch::nn::Conv2d detect4{nullptr};
};
TORCH_MODULE(Head);

#endif