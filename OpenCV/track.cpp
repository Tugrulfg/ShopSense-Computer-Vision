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
#include <chrono>
#include <filesystem>
#include<tuple>
#include <array>
#include "omp.h"
#include <cmath>
#include "Tracker.hpp"

void getBoxes(const cv::Mat& label_matrix, size_t grid, std::vector<std::pair<cv::Rect2f, int>>& detections);

float sigmoid(float in);

const std::array<const char*, 10> CLASSES = {"apple", "centro", "chips", "drain_opener", "ketchup", "pasta", "potato", "probis", "semolina", "tea"};
const size_t IMAGE_SIZE = 224;
const size_t GRID_SIZE = 7;
const size_t NUM_BOXES = 2;
const size_t NUM_CLASSES = 10;
const float OBJ_THRESHOLD = 0.45;        // Ignore the detections with confidence less than threshold
const float NMS_THRESHOLD = 0.05;        // Ignore the detections with IOU more than threshold

std::vector<std::string> getFiles(const std::string& path);

int main(){
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    std::vector<std::pair<int, int>> results;
    MyTracker tracker;
    cv::dnn::Net net = cv::dnn::readNetFromONNX("../Models/YOLOTiny_LargeNOOBJ/yolo_tiny_op11.onnx");
    std::vector<std::pair<cv::Rect2f, int>> detections;
    int sz[3]= {7, 7, 20}; 


    cv::Mat frame, input, output;

    cv::VideoCapture cap("../TestVideo/20240528_164634.mp4");
    // cv::VideoCapture cap(0);
    std::vector<std::string> test_images = getFiles("../TestImages/");
    std::cout << test_images.size() << std::endl;
    //for(const std::string& path: test_images){
    while(1){
        cv::TickMeter tm;
        tm.start();
        cap >> frame;

        if (frame.empty()) {
            std::cerr << "Error: Unable to read a frame from the video." << std::endl;
            break;
        }
        // frame = cv::imread(path);
        cv::resize(frame, frame, cv::Size(IMAGE_SIZE, IMAGE_SIZE));
        cv::dnn::blobFromImage(frame, input, 1.0/255.0, cv::Size(224, 224), cv::Scalar(), true, false);
        net.setInput(input);
        //std::vector<std::string> outputNames = net.getUnconnectedOutLayersNames();

        // Create a vector to store the outputs
        // std::vector<cv::Mat> outputs;

        // Run the forward pass and retrieve the outputs
        output = net.forward();
        cv::Mat newmat(3, sz, output.type(), output.ptr<float>(0));
        
        getBoxes(newmat, 7, detections);
        //Process the outputs
        // for (size_t i = 0; i < outputs.size(); ++i) {
        //     std::cout << outputs[i].size << std::endl;
        //     cv::Mat newmat(3, sz[i], outputs[i].type(), outputs[i].ptr<float>(0));
        //     getBoxes(newmat, GRIDS[i], detections);
        // }
        

        results = tracker.update(detections);
        for(const std::pair<int, int>& result : results){
            if(result.first == 0)
                std::cout << "BOTTOM: " << CLASSES[result.second] << std::endl;
            else
                std::cout << "TOP: " << CLASSES[result.second] << std::endl;
        }
        tracker.draw(frame);

        // for(const std::pair<cv::Rect2f, int>& detection: detections){
        //     cv::putText(frame, CLASSES[detection.second], detection.first.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 2);
        //     cv::rectangle(frame, detection.first, cv::Scalar(0, 255, 255), 2);
        // }
        // double fps = cv::getTickFrequency() / (end - start);
        // std::cout << fps << std::endl;
        // cv::putText(frame, std::to_string(fps), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        tm.stop();
        double inferenceTime = tm.getTimeMilli(); // In milliseconds
        double fps = 1000.0 / inferenceTime;
        cv::putText(frame, std::to_string(fps), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        cv::imshow("frame", frame);
        if(cv::waitKey(1) == 'q')
            break;

        
        detections.clear();
        results.clear();
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}

void getBoxes(const cv::Mat& label_matrix, size_t grid, std::vector<std::pair<cv::Rect2f, int>>& detections) {
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    for (int i = 0; i < 7; ++i) {
        for(int j=0; j<7; j++){
            if(label_matrix.at<float>(i, j, 10) > label_matrix.at<float>(i, j, 15)){
                float confidence = label_matrix.at<float>(i, j, 10);
                confidence = sigmoid(confidence);
                if (confidence > OBJ_THRESHOLD) {
                    int centerX = static_cast<int>((label_matrix.at<float>(i, j, 11) + j) * (IMAGE_SIZE/grid));
                    int centerY = static_cast<int>((label_matrix.at<float>(i, j, 12) + i) * (IMAGE_SIZE/grid));
                    int width = static_cast<int>(label_matrix.at<float>(i, j, 13) * IMAGE_SIZE);
                    int height = static_cast<int>(label_matrix.at<float>(i, j, 14) * IMAGE_SIZE);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    int maxIndex = 0;
                    float maxValue = label_matrix.at<float>(i, j, 0); // Initialize maxValue with the first element
                    for (int k = 1; k < 10; ++k) {
                        float currentValue = label_matrix.at<float>(i, j, k);
                        if (currentValue > maxValue) {
                            maxValue = currentValue;
                            maxIndex = k;
                        }
                    }
                    classIds.push_back(maxIndex);
                    confidences.push_back(confidence);
                    boxes.push_back(cv::Rect(left, top, width, height));
                }  
            }
            else{
                float confidence = label_matrix.at<float>(i, j, 15);
                confidence = sigmoid(confidence);
                if (confidence > OBJ_THRESHOLD) {
                    int centerX = static_cast<int>((label_matrix.at<float>(i, j, 16) + j) * (IMAGE_SIZE/grid));
                    int centerY = static_cast<int>((label_matrix.at<float>(i, j, 17) + i) * (IMAGE_SIZE/grid));
                    int width = static_cast<int>(label_matrix.at<float>(i, j, 18) * IMAGE_SIZE);
                    int height = static_cast<int>(label_matrix.at<float>(i, j, 19) * IMAGE_SIZE);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    int maxIndex = 0;
                    float maxValue = label_matrix.at<float>(i, j, 0); // Initialize maxValue with the first element
                    for (int k = 1; k < 10; ++k) {
                        float currentValue = label_matrix.at<float>(i, j, k);
                        if (currentValue > maxValue) {
                            maxValue = currentValue;
                            maxIndex = k;
                        }
                    }
                    classIds.push_back(maxIndex);
                    confidences.push_back(confidence);
                    boxes.push_back(cv::Rect(left, top, width, height));
                }  
            }
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, OBJ_THRESHOLD, NMS_THRESHOLD, indices);

    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        detections.push_back({boxes[idx], classIds[idx]});
    }
}

float sigmoid(float in) {
    return 1.0 / (1.0 + std::exp(-in));
}

std::vector<std::string> getFiles(const std::string& path){
    std::vector<std::string> filenames;
    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        if (entry.is_regular_file()) {
            std::string fileName = entry.path().filename().string();
            filenames.push_back(path + fileName);
        }
    }
    return filenames;
}