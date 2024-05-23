#include <iostream>
#include "opencv2/opencv.hpp"
#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <vector>
#include <fstream>
#include <dirent.h>
#include <string>
#include <filesystem>
#include<tuple>
#include <array>
#include "omp.h"

// struct Detection {
//     cv::Rect2f box;
//     int class_id;
//     float score;
// };

// std::vector<Detection> getBoxes(const cv::Mat& label_matrix);

// std::vector<std::pair<cv::Rect2f, int>> NMS(const torch::Tensor& labels);

// torch::Tensor intersection_over_union(const torch::Tensor& boxes_preds, const torch::Tensor& boxes_labels);

const std::array<const char*, 10> CLASSES = {"apple", "centro", "chips", "drain_opener", "ketchup", "pasta", "potato", "probis", "semolina", "tea"};
const size_t IMAGE_SIZE = 224;
const size_t GRID_SIZE = 7;
const size_t NUM_BOXES = 2;
const size_t NUM_CLASSES = 10;
const float OBJ_THRESHOLD = 1.0;        // Ignore the detections with confidence less than threshold
const float NMS_THRESHOLD = 0.45;        // Ignore the detections with IOU more than threshold

int main(){
    cv::dnn::Net net = cv::dnn::readNetFromONNX("../Models/model.onnx");
    std::vector<Detection> detections;

    cv::Mat frame, input, output;

    cv::VideoCapture cap(0);
    int start, end;
    while(1){
        start = cv::getTickCount();
        cap >> frame;

        input = cv::dnn::blobFromImage(frame, 1 / 127.5, cv::Size(IMAGE_SIZE, IMAGE_SIZE), cv::Scalar(127.5, 127.5, 127.5), true, false);
        net.setInput(input);
        output = net.forward();
        //detections = getBoxes(output);

        //std::cout << detections.size() << std::endl;
        end = cv::getTickCount();
        double fps = cv::getTickFrequency() / (end - start);
        std::cout << fps << std::endl;

        //cv::imshow("frame", frame);
        if(cv::waitKey(1) == 'q')
            break;
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}

// std::vector<Detection> getBoxes(const cv::Mat& label_matrix){
//     std::vector<Detection> detections;
    
//     int stride = GRID_SIZE * GRID_SIZE;
//     const float* data = reinterpret_cast<const float*>(label_matrix.data);

//     for (int cy = 0; cy < GRID_SIZE; ++cy) {
//         for (int cx = 0; cx < GRID_SIZE; ++cx) {
//             int index = cy * GRID_SIZE + cx;

//             // Get class scores and objectness score
//             float objectness = data[index + 4 * stride];
//             //std::cout << objectness << std::endl;
//             if (objectness < OBJ_THRESHOLD) {
//                 continue;  // Skip low confidence detections
//             }

//             float class_scores[NUM_CLASSES];
//             for (int i = 0; i < NUM_CLASSES; ++i) {
//                 class_scores[i] = data[index + (5 + i) * stride];
//             }

//             // Get the best class and its score
//             int class_id = std::max_element(class_scores, class_scores + NUM_CLASSES) - class_scores;
//             float class_score = class_scores[class_id];
//             float score = objectness * class_score;

//             // Get bounding box coordinates
//             float bx = data[index] * IMAGE_SIZE;
//             float by = data[index + stride] * IMAGE_SIZE;
//             float bw = data[index + 2 * stride] * IMAGE_SIZE;
//             float bh = data[index + 3 * stride] * IMAGE_SIZE;

//             // Convert from center coordinates to top-left coordinates
//             float x = bx - bw / 2;
//             float y = by - bh / 2;

//             detections.push_back({cv::Rect2f(x, y, bw, bh), class_id, score});
//         }
//     }
//     return std::move(detections);
//     // torch::Tensor bboxes;
//     // torch::Tensor scores;
//     // torch::Tensor classes;

//     // torch::Tensor img_size = torch::zeros({1}).fill_(static_cast<double>(IMAGE_SIZE));
//     // torch::Tensor grid_size = torch::zeros({1}).fill_(static_cast<double>(GRID_SIZE));

//     // torch::Tensor bestbox;
//     // torch::Tensor cat_scores = torch::cat({label_matrix.slice(-1, 10, 11), label_matrix.slice(-1, 15, 16)}, /*dim=*/0);

//     // auto res = torch::max(cat_scores, /*dim=*/0);
//     // bestbox = std::get<1>(res);
//     // scores = torch::sigmoid(std::get<0>(res).view({-1, 1}));
//     // classes = label_matrix.slice(-1, 0, 10).argmax(-1).view({-1, 1});
//     // bboxes =(bestbox * label_matrix.slice(/*dim=*/-1, /*start=*/16, /*end=*/20) + (1 - bestbox) * label_matrix.slice(/*dim=*/-1, /*start=*/11, /*end=*/15));

//     // torch::Tensor cell_indices = torch::arange((int)GRID_SIZE).repeat({1, (int)GRID_SIZE, 1}).unsqueeze(-1);
//     // torch::Tensor w = img_size * bboxes.slice(-1, 2, 3).clamp(0);
//     // torch::Tensor h = img_size * bboxes.slice(-1, 3, 4).clamp(0);
//     // torch::Tensor x = (img_size / grid_size) * ((bboxes.slice(-1, 0, 1).clamp(0) + cell_indices));
//     // torch::Tensor y = (img_size / grid_size) * ((bboxes.slice(-1, 1, 2).clamp(0) + cell_indices.permute({0, 2, 1, 3})));

//     // torch::Tensor converted_bboxes = torch::cat({(x - w / 2).clamp(0), (y - h / 2).clamp(0), w, h}, -1).view({-1, 4});

//     // torch::Tensor labels = torch::cat({classes, scores, converted_bboxes}, /*dim=*/-1);

//     // // Confidence threshold
//     // torch::Tensor indices = torch::nonzero(labels.slice(/*dim=*/-1, /*start=*/1, /*end=*/2).flatten() > OBJ_THRESHOLD).squeeze(-1);

//     // return torch::index_select(labels, /*dim=*/0, indices);
// }

// std::vector<std::pair<cv::Rect2f, int>> NMS(const torch::Tensor& labels){
//     torch::Tensor order = torch::argsort(labels.slice(/*dim=*/-1, /*start=*/1, /*end=*/2), /*dim=*/0, /*descending=*/true);
//     std::vector<torch::Tensor> ordered_boxes;
//     std::vector<std::pair<cv::Rect2f, int>> final_boxes;
//     torch::Tensor chosen_box;

//     for(size_t i=0; i<order.sizes()[0]; i++){
//         ordered_boxes.push_back(labels[order[i].item<int>()]);
//     }

//     while(ordered_boxes.size() > 0){
//         chosen_box = ordered_boxes[0];
//         final_boxes.push_back({cv::Rect2f(chosen_box[2].item<float>(), chosen_box[3].item<float>(), chosen_box[4].item<float>(), chosen_box[5].item<float>()), chosen_box[0].item<int>()});
//         ordered_boxes.erase(ordered_boxes.begin());
//         for(size_t i=0; i<ordered_boxes.size(); i++){
//             if(torch::equal(chosen_box[0], ordered_boxes[i][0]) && torch::gt(intersection_over_union(chosen_box.slice(/*dim=*/-1, /*start=*/2, /*end=*/6), ordered_boxes[i].slice(/*dim=*/-1, /*start=*/2, /*end=*/6)), torch::tensor(NMS_THRESHOLD)).item<bool>()){
//                 ordered_boxes.erase(ordered_boxes.begin() + i);
//                 i--;
//             }
//         }
//     }

//     return final_boxes;
// }

// torch::Tensor intersection_over_union(const torch::Tensor& boxes_preds, const torch::Tensor& boxes_labels){
//     torch::Tensor box1_x = boxes_preds.slice(/*dim=*/-1, /*start=*/0, /*end=*/1);
//     torch::Tensor box1_y = boxes_preds.slice(/*dim=*/-1, /*start=*/1, /*end=*/2);
//     torch::Tensor box1_w = boxes_preds.slice(/*dim=*/-1, /*start=*/2, /*end=*/3);
//     torch::Tensor box1_h = boxes_preds.slice(/*dim=*/-1, /*start=*/3, /*end=*/4);
//     torch::Tensor box2_x = boxes_labels.slice(/*dim=*/-1, /*start=*/0, /*end=*/1);
//     torch::Tensor box2_y = boxes_labels.slice(/*dim=*/-1, /*start=*/1, /*end=*/2);
//     torch::Tensor box2_w = boxes_labels.slice(/*dim=*/-1, /*start=*/2, /*end=*/3);
//     torch::Tensor box2_h = boxes_labels.slice(/*dim=*/-1, /*start=*/3, /*end=*/4);

//     torch::Tensor intersection = (torch::min(box1_x+box1_w, box2_x+box2_w) - torch::max(box1_x, box2_x)).clamp(0) * (torch::min(box1_y+box1_h, box2_y+box2_h - torch::max(box1_y, box2_y)).clamp(0));

//     return intersection / (torch::abs(box1_w * box1_h) + torch::abs(box2_w * box2_h) - intersection + 1e-7);
// }
