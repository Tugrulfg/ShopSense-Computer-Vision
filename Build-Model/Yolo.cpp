#include "Yolo.hpp"
#include<fstream>

YOLOImpl::YOLOImpl(const int num_boxes, const int num_classes, const int grid_size): NUM_BOXES(num_boxes), NUM_CLASSES(num_classes), GRID_SIZE(grid_size){
    backbone = register_module("backbone", Backbone());
    head = register_module("head", Head(num_boxes, num_classes, grid_size));
}

torch::Tensor YOLOImpl::forward(torch::Tensor x) {
    x = backbone->forward(x);
    return head->forward(x);
}

size_t YOLOImpl::numParameters() const{
    size_t total_parameters = 0;
    for (auto& parameter : this->parameters()) {
        total_parameters += parameter.numel();
    }
    return total_parameters;
}

void YOLOImpl::save_weights(const std::string& filename) const{
    std::ofstream file(filename);
    file <<"{ " << "\n";
    size_t layer = 1;
    // Iterate over each parameter in the model
    for (auto& p : this->named_parameters()) {
        std::cout << layer++ << ". Layer" << std::endl;
        if (p.value().is_cuda()) {
            p.value() = p.value().cpu();
        }

        // Write the parameter name and weights to the file
        file << '"' << p.key() << '"' << ": [";

        if(p.value().sizes().size() == 4){              // Conv2D layer weights
            for(size_t i=0; i < p.value().sizes()[0]; i++){
                file << "[";
                for(size_t j=0; j < p.value().sizes()[1]; j++){
                    file << "[";
                    for(size_t k=0; k < p.value().sizes()[2]; k++){
                        file << "[";
                        for(size_t l=0; l < p.value().sizes()[3]; l++){
                            if(l == p.value().sizes()[3]-1){
                                file << p.value()[i][j][k][l].item<float>();
                                break;
                            }
                            file << p.value()[i][j][k][l].item<float>() << ", ";
                        }
                        if(k == p.value().sizes()[2]-1){
                            file << "]";
                            break;
                        }
                        file << "],\n";
                    }
                    if(j == p.value().sizes()[1]-1){
                        file << "]";
                        break;
                    }
                    file << "],\n";
                }
                if(i == p.value().sizes()[0]-1){
                    file << "]";
                    break;
                }
                file << "],\n";
            }
        }
        else if(p.value().sizes().size() == 1){         // Biases
            for(size_t i=0; i < p.value().sizes()[0]; i++){
                if(i == p.value().sizes()[0]-1){
                    file << p.value()[i].item<float>();
                    break;
                }
                file << p.value()[i].item<float>() << ", ";
            }
        }
        else{                                           // Linear Layer weights
            for(size_t i=0; i < p.value().sizes()[0]; i++){
                file << "[";
                for(size_t j=0; j < p.value().sizes()[1]; j++){
                    if(j == p.value().sizes()[1]-1){
                        file << p.value()[i][j].item<float>();
                        break;
                    }
                    file << p.value()[i][j].item<float>() << ", ";
                }
                if(i == p.value().sizes()[0]-1){
                    file << "]";
                    break;
                }
                file << "],\n";
            }
        }

        if (torch::equal(*p, this->parameters().back())) {      // End of the file
            file << "]\n";
            break;
        }
        else
            file << "],\n";
    }
    file << "}" << "\n";


    file.close();
}

void YOLOImpl::load_pretrained(const std::string& filename){
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    std::string buffer;
    std::vector<double> tokens;

    // Iterate through the model parameters and assign weights from JSON
    for (auto& p : backbone->named_parameters()) {
        std::cout << "Target:" << p.value().sizes() << std::endl;

        int sizes = p.value().sizes().size();
        int size1, size2, size3, size4;

        if(sizes == 4){
            size1 = p.value().sizes()[0];
            size2 = p.value().sizes()[1];
            size3 = p.value().sizes()[2];
            size4 = p.value().sizes()[3];
        }
        else if(sizes == 1){
            size1 = p.value().sizes()[0];
        }
        else{
            std::cout << "ERROR SIZE:" << sizes << std::endl;
            return;
        }

        std::getline(ifs, buffer);
        std::stringstream ss(buffer);

        double num;
        while(ss >> num)
            tokens.push_back(num);

        std::cout << "Tokens: " << tokens.size() << std::endl;
        torch::Tensor tensor = torch::tensor(tokens);
        if(sizes == 4)
            p.value().detach().copy_(tensor.view({size1, size2, size3, size4}));
        else if(sizes == 1){
            p.value().detach().copy_(tensor.view({size1}));
        }
        else{
            std::cout << "ERROR SIZE:" << sizes << std::endl;
            return;
        }

        tokens.clear();
        std::cout << "END" << std::endl;
    }
}