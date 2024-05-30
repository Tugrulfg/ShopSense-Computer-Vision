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
const float OBJ_THRESHOLD = 0.30;        // Ignore the detections with confidence less than threshold
const float NMS_THRESHOLD = 0.00;        // Ignore the detections with IOU more than threshold

int main(){
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    std::vector<std::pair<int, int>> results;
    int classes[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // Initialize the tracker
    MyTracker tracker("../Models/final_model_op10.onnx", NMS_THRESHOLD, OBJ_THRESHOLD);
    cv::Mat frame;

    cv::VideoCapture cap(0);
    while(1){
        // cv::TickMeter tm;
        // tm.start();
        cap >> frame;

        if (frame.empty()) {
            std::cerr << "Error: Unable to read a frame from the video." << std::endl;
            break;
        }
        cv::resize(frame, frame, cv::Size(224, 224));
        results = tracker.update(frame);
        
        for(const std::pair<int, int>& result : results){
            if(classes[result.second] == 1)
                continue;
            if(result.first == 0)
                std::cout << "BOTTOM: " << CLASSES[result.second] << std::endl;
            else if(result.first == 1)
                std::cout << "TOP: " << CLASSES[result.second] << std::endl;
            classes[result.second] = 1;
        }
        tracker.draw(frame);

        // tm.stop();
        // double inferenceTime = tm.getTimeMilli(); // In milliseconds
        // double fps = 1000.0 / inferenceTime;
        // cv::putText(frame, std::to_string(fps), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        cv::imshow("frame", frame);

        if(cv::waitKey(1) == 'q')
            break;

        for(int i=0; i<10; i++)
            classes[i] = 0;
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}