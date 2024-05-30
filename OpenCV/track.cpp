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
#include <array>
#include "omp.h"
#include "Tracker.hpp"

const std::array<const char*, 10> CLASSES = {"apple", "centro", "chips", "drain_opener", "ketchup", "pasta", "potato", "probis", "semolina", "tea"};
const float OBJ_THRESHOLD = 0.55;        // Ignore the detections with confidence less than threshold
const float NMS_THRESHOLD = 0.45;        // Ignore the detections with IOU more than threshold

int main(){
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    std::vector<std::pair<int, int>> results;
    MyTracker tracker("../Models/yolo_tiny_op10.onnx", NMS_THRESHOLD, OBJ_THRESHOLD);
    std::vector<std::pair<cv::Rect2f, int>> detections;
    int sz[3]= {7, 7, 20}; 

    cv::Mat frame, input, output;

    cv::VideoCapture cap("../TestVideo/20240528_164634.mp4");
    // cv::VideoCapture cap(0);
    while(1){
        cv::TickMeter tm;
        tm.start();
        cap >> frame;

        if (frame.empty()) {
            std::cerr << "Error: Unable to read a frame from the video." << std::endl;
            break;
        }
        cv::resize(frame, frame, cv::Size(224, 224));
        results = tracker.update(frame);
        
        for(const std::pair<int, int>& result : results){
            if(result.first == 0)
                std::cout << "BOTTOM: " << CLASSES[result.second] << std::endl;
            else if(result.first == 1)
                std::cout << "TOP: " << CLASSES[result.second] << std::endl;
        }
        tracker.draw(frame);

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