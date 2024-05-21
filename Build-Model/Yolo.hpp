#ifndef YOLO_HPP
#define YOLO_HPP

#include <torch/torch.h>
#include <torch/script.h>
#include "Layers.hpp"
#include <string>

// YOLO model
class YOLOImpl: public torch::nn::Module{
    public:
        YOLOImpl(const int num_boxes, const int num_classes, const int grid_size);

        // Takes input image and returns the predictions
        torch::Tensor forward(torch::Tensor x);

        // Returns the number of learnable parameters
        size_t numParameters() const;

        // Saves all the weights in .json format
        void save_weights(const std::string& filename) const;

        void load_pretrained(const std::string& filename);
    private:
        // Parameters about the model
        const int NUM_BOXES, NUM_CLASSES, GRID_SIZE;

        // Modules that are used in the model
        Backbone backbone{nullptr};
        Head head{nullptr};
};
TORCH_MODULE(YOLO);

#endif