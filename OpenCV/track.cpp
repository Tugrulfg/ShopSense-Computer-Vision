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
#include <cmath>
#include "Tracker.hpp"

std::vector<std::pair<cv::Rect2f, int>> getBoxes(const cv::Mat& label_matrix);

float sigmoid(float in);

const std::array<const char*, 10> CLASSES = {"apple", "centro", "chips", "drain_opener", "ketchup", "pasta", "potato", "probis", "semolina", "tea"};
const size_t IMAGE_SIZE = 224;
const size_t GRID_SIZE = 7;
const size_t NUM_BOXES = 2;
const size_t NUM_CLASSES = 10;
const float OBJ_THRESHOLD = 0.45;        // Ignore the detections with confidence less than threshold
const float NMS_THRESHOLD = 0.45;        // Ignore the detections with IOU more than threshold

int main(){
    Tracker tracker;
    cv::dnn::Net net = cv::dnn::readNetFromONNX("../Models/model.onnx");
    std::vector<std::pair<cv::Rect2f, int>> detections;
    int sz[] = {7, 7, 20}; 


    cv::Mat frame, input, output;

    cv::VideoCapture cap(0);
    int start, end;
    while(1){
        start = cv::getTickCount();
        cap >> frame;

        input = cv::dnn::blobFromImage(frame, 1 / 127.5, cv::Size(IMAGE_SIZE, IMAGE_SIZE), cv::Scalar(127.5, 127.5, 127.5), true, false);
        net.setInput(input);
        output = net.forward();
        cv::Mat newmat(3, sz, output.type(), output.ptr<float>(0));

        detections = getBoxes(newmat);

        tracker.update(detections);
        //tracker.draw(frame);
        end = cv::getTickCount();
        double fps = cv::getTickFrequency() / (end - start);
        std::cout << fps << std::endl;
        // cv::putText(frame, std::to_string(fps), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        // cv::imshow("frame", frame);
        // if(cv::waitKey(1) == 'q')
        //     break;

        detections.clear();
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}

std::vector<std::pair<cv::Rect2f, int>> getBoxes(const cv::Mat& label_matrix) {
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    for (int i = 0; i < 7; ++i) {
        for(int j=0; j<7; j++){
            if(label_matrix.at<float>(i, j, 10) > label_matrix.at<float>(i, j, 15)){
                float confidence = label_matrix.at<float>(i, j, 10);
                confidence = sigmoid(confidence);
                if (confidence > OBJ_THRESHOLD) {
                    int centerX = static_cast<int>((label_matrix.at<float>(i, j, 11) + j) * (IMAGE_SIZE/GRID_SIZE));
                    int centerY = static_cast<int>((label_matrix.at<float>(i, j, 12) + i) * (IMAGE_SIZE/GRID_SIZE));
                    int width = static_cast<int>(label_matrix.at<float>(i, j, 13) * IMAGE_SIZE);
                    int height = static_cast<int>(label_matrix.at<float>(i, j, 14) * IMAGE_SIZE);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    int maxIndex = 0;
                    float maxValue = label_matrix.at<float>(0, 0, 0); // Initialize maxValue with the first element
                    for (int i = 1; i < 10; ++i) {
                        float currentValue = label_matrix.at<float>(0, 0, i);
                        if (currentValue > maxValue) {
                            maxValue = currentValue;
                            maxIndex = i;
                        }
                    }
                    classIds.push_back(label_matrix.at<float>(i, j, maxIndex));
                    confidences.push_back(confidence);
                    boxes.push_back(cv::Rect(left, top, width, height));
                }  
            }
            else{
                float confidence = label_matrix.at<float>(i, j, 15);
                confidence = sigmoid(confidence);
                if (confidence > OBJ_THRESHOLD) {
                    int centerX = static_cast<int>((label_matrix.at<float>(i, j, 16) + j) * (IMAGE_SIZE/GRID_SIZE));
                    int centerY = static_cast<int>((label_matrix.at<float>(i, j, 17) + i) * (IMAGE_SIZE/GRID_SIZE));
                    int width = static_cast<int>(label_matrix.at<float>(i, j, 18) * IMAGE_SIZE);
                    int height = static_cast<int>(label_matrix.at<float>(i, j, 19) * IMAGE_SIZE);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    int maxIndex = 0;
                    float maxValue = label_matrix.at<float>(0, 0, 0); // Initialize maxValue with the first element
                    for (int i = 1; i < 10; ++i) {
                        float currentValue = label_matrix.at<float>(0, 0, i);
                        if (currentValue > maxValue) {
                            maxValue = currentValue;
                            maxIndex = i;
                        }
                    }
                    classIds.push_back(label_matrix.at<float>(i, j, maxIndex));
                    confidences.push_back(confidence);
                    boxes.push_back(cv::Rect(left, top, width, height));
                }  
            }
             
        }
    }

    std::vector<std::pair<cv::Rect2f, int>> result;
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, OBJ_THRESHOLD, NMS_THRESHOLD, indices);

    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        result.push_back({boxes[idx], classIds[idx]});
    }

    return result;
}

float sigmoid(float in) {
    return 1.0 / (1.0 + std::exp(-in));
}