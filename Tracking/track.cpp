#include <iostream>
#include "opencv2/opencv.hpp"


#include "../Build-Model/Yolo.hpp"
#include <vector>
#include <fstream>
#include <dirent.h>
#include <string>
#include <filesystem>
#include<tuple>
#include <array>
#include "Tracker.hpp"
#include "omp.h"

torch::Tensor ImagetoTensor(const cv::Mat& img);

torch::Tensor getBoxes(const torch::Tensor& label_matrix);

std::vector<std::pair<cv::Rect2f, int>> NMS(const torch::Tensor& labels);

torch::Tensor intersection_over_union(const torch::Tensor& boxes_preds, const torch::Tensor& boxes_labels);

const std::array<const char*, 10> CLASSES = {"apple", "centro", "chips", "drain_opener", "ketchup", "pasta", "potato", "probis", "semolina", "tea"};
const size_t IMAGE_SIZE = 224;
const size_t GRID_SIZE = 7;
const size_t NUM_BOXES = 2;
const size_t NUM_CLASSES = 10;
const float OBJ_THRESHOLD = 0.45;        // Ignore the detections with confidence less than threshold
const float NMS_THRESHOLD = 0.45;        // Ignore the detections with IOU more than threshold

int main(){
    std::vector<std::pair<int, int>> results;
    Tracker tracker;
    YOLO model = YOLO(NUM_BOXES, NUM_CLASSES, GRID_SIZE);
    torch::load(model, "../Models/model.pt");

    model->to(torch::kCPU);
    model->eval();

    cv::Mat image;
    torch::Tensor input_frame;
    torch::Tensor output;
    torch::Tensor boxes;
    std::vector<std::pair<cv::Rect2f, int>> labels;

    cv::Point leftTop, rightTop;
    double fps = 0;
    int64 start = 0;

	cv::VideoCapture cap(0);
    while(1){
        start = cv::getTickCount();
        cap >> image;

        cv::resize(image, image, cv::Size{IMAGE_SIZE, IMAGE_SIZE}, 0, 0, cv::INTER_LINEAR);
        input_frame = ImagetoTensor(image).view({1, 3, IMAGE_SIZE, IMAGE_SIZE});
        
        output = model->forward(input_frame);       
        
        boxes = getBoxes(output);

        labels = NMS(boxes);

        results = tracker.update(labels);
        for(const std::pair<int, int>& result : results){
            if(result.first == 0)
                std::cout << "BOTTOM: " << CLASSES[result.second] << std::endl;
            else
                std::cout << "TOP: " << CLASSES[result.second] << std::endl;
        }
        //tracker.draw(image);

        
        fps = cv::getTickFrequency() / (cv::getTickCount() - start);
        std::cout << fps << std::endl;
        //cv::putText(image, "FPS : " + std::to_string(fps), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

        //cv::imshow("image", image);
        
        // if(cv::waitKey(1) == 'q')           // Stop when pressed 'q'
		//     break;
            
        labels.clear();
        results.clear();
    }

    cap.release();

    return 0;
}

torch::Tensor ImagetoTensor(const cv::Mat& img){
    cv::Mat cpy;
    cv::cvtColor(img, cpy, cv::COLOR_BGR2RGB);
    torch::Tensor img_tensor = torch::from_blob(cpy.data, {IMAGE_SIZE, IMAGE_SIZE, 3}, torch::kByte);
    img_tensor = img_tensor.permute({2, 0, 1}).toType(torch::kFloat);
    return img_tensor / 127.5 - 1.0;
}

torch::Tensor getBoxes(const torch::Tensor& label_matrix){
    torch::Tensor bboxes;
    torch::Tensor scores;
    torch::Tensor classes;

    torch::Tensor img_size = torch::zeros({1}).fill_(static_cast<double>(IMAGE_SIZE));
    torch::Tensor grid_size = torch::zeros({1}).fill_(static_cast<double>(GRID_SIZE));

    torch::Tensor bestbox;
    torch::Tensor cat_scores = torch::cat({label_matrix.slice(-1, 10, 11), label_matrix.slice(-1, 15, 16)}, /*dim=*/0);

    auto res = torch::max(cat_scores, /*dim=*/0);
    bestbox = std::get<1>(res);
    scores = torch::sigmoid(std::get<0>(res).view({-1, 1}));
    classes = label_matrix.slice(-1, 0, 10).argmax(-1).view({-1, 1});
    bboxes =(bestbox * label_matrix.slice(/*dim=*/-1, /*start=*/16, /*end=*/20) + (1 - bestbox) * label_matrix.slice(/*dim=*/-1, /*start=*/11, /*end=*/15));

    torch::Tensor cell_indices = torch::arange((int)GRID_SIZE).repeat({1, (int)GRID_SIZE, 1}).unsqueeze(-1);
    torch::Tensor w = img_size * bboxes.slice(-1, 2, 3).clamp(0);q
    torch::Tensor h = img_size * bboxes.slice(-1, 3, 4).clamp(0);
    torch::Tensor x = (img_size / grid_size) * ((bboxes.slice(-1, 0, 1).clamp(0) + cell_indices));
    torch::Tensor y = (img_size / grid_size) * ((bboxes.slice(-1, 1, 2).clamp(0) + cell_indices.permute({0, 2, 1, 3})));

    torch::Tensor converted_bboxes = torch::cat({(x - w / 2).clamp(0), (y - h / 2).clamp(0), w, h}, -1).view({-1, 4});

    torch::Tensor labels = torch::cat({classes, scores, converted_bboxes}, /*dim=*/-1);

    // Confidence threshold
    torch::Tensor indices = torch::nonzero(labels.slice(/*dim=*/-1, /*start=*/1, /*end=*/2).flatten() > OBJ_THRESHOLD).squeeze(-1);

    return torch::index_select(labels, /*dim=*/0, indices);
}

std::vector<std::pair<cv::Rect2f, int>> NMS(const torch::Tensor& labels){
    torch::Tensor order = torch::argsort(labels.slice(/*dim=*/-1, /*start=*/1, /*end=*/2), /*dim=*/0, /*descending=*/true);
    std::vector<torch::Tensor> ordered_boxes;
    std::vector<std::pair<cv::Rect2f, int>> final_boxes;
    torch::Tensor chosen_box;

    for(size_t i=0; i<order.sizes()[0]; i++){
        ordered_boxes.push_back(labels[order[i].item<int>()]);
    }

    while(ordered_boxes.size() > 0){
        chosen_box = ordered_boxes[0];
        final_boxes.push_back({cv::Rect2f(chosen_box[2].item<float>(), chosen_box[3].item<float>(), chosen_box[4].item<float>(), chosen_box[5].item<float>()), chosen_box[0].item<int>()});
        ordered_boxes.erase(ordered_boxes.begin());
        for(size_t i=0; i<ordered_boxes.size(); i++){
            if(torch::equal(chosen_box[0], ordered_boxes[i][0]) && torch::gt(intersection_over_union(chosen_box.slice(/*dim=*/-1, /*start=*/2, /*end=*/6), ordered_boxes[i].slice(/*dim=*/-1, /*start=*/2, /*end=*/6)), torch::tensor(NMS_THRESHOLD)).item<bool>()){
                ordered_boxes.erase(ordered_boxes.begin() + i);
                i--;
            }
        }
    }

    return final_boxes;
}

torch::Tensor intersection_over_union(const torch::Tensor& boxes_preds, const torch::Tensor& boxes_labels){
    torch::Tensor box1_x = boxes_preds.slice(/*dim=*/-1, /*start=*/0, /*end=*/1);
    torch::Tensor box1_y = boxes_preds.slice(/*dim=*/-1, /*start=*/1, /*end=*/2);
    torch::Tensor box1_w = boxes_preds.slice(/*dim=*/-1, /*start=*/2, /*end=*/3);
    torch::Tensor box1_h = boxes_preds.slice(/*dim=*/-1, /*start=*/3, /*end=*/4);
    torch::Tensor box2_x = boxes_labels.slice(/*dim=*/-1, /*start=*/0, /*end=*/1);
    torch::Tensor box2_y = boxes_labels.slice(/*dim=*/-1, /*start=*/1, /*end=*/2);
    torch::Tensor box2_w = boxes_labels.slice(/*dim=*/-1, /*start=*/2, /*end=*/3);
    torch::Tensor box2_h = boxes_labels.slice(/*dim=*/-1, /*start=*/3, /*end=*/4);

    torch::Tensor intersection = (torch::min(box1_x+box1_w, box2_x+box2_w) - torch::max(box1_x, box2_x)).clamp(0) * (torch::min(box1_y+box1_h, box2_y+box2_h - torch::max(box1_y, box2_y)).clamp(0));

    return intersection / (torch::abs(box1_w * box1_h) + torch::abs(box2_w * box2_h) - intersection + 1e-7);
}
