#ifndef TRACKER_HPP
#define TRACKER_HPP

#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include<array>
#include <deque>
#include <unordered_map>
#include <numeric>
#include <cmath>

// Class for storing tracked object state
class KalmanFilter {
    public:
        KalmanFilter(const cv::Rect2f& init_bbox, int class_id);

        // Predicts the next state of the object based on the current state
        void predict();

        // Updates the state of the object based on the measurement
        void update(const cv::Rect2f& bbox, int class_id);

        // Returns the current location of the object
        cv::Rect2f get_bbox() const;

        // Returns the class of the object
        int get_class_id() const;

        // Returns the time since the object was last updated
        int get_time_since_update() const;

        // Increments the time since the object was last updated
        void increment_time_since_update();

        // Updates the class of the object
        void update_class(int new_class);

        // Returns the number of times the object has been tracked
        int get_hits() const;

        // Returns the exit direction of the object: 0-->Bottom, 1-->Top
        int exit_dir()const;        

    private:
        cv::KalmanFilter kf;
        cv::Mat state;
        cv::Mat meas;
        int class_id;
        std::deque<int> classes;
        int age;
        int hits;
        int time_since_update;
        int enter_dir;
};

// Class encapsulating the detection model and tracking algorithm
class MyTracker {
    public:
        MyTracker(const std::string model_path, const float nms_thres, const float obj_thres);
        ~MyTracker();

        // Takes the input frame and updates the tracked objects
        std::vector<std::pair<int, int>> update(const cv::Mat& frame);

        // Draws the detections to the frame
        void draw(cv::Mat& frame)const;

    private:
        cv::dnn::Net model;
        std::vector<KalmanFilter> trackers;
        int max_age = 10; // Maximum number of frames to keep a tracker without updates
        int min_hits = 2; // Minimum number of hits to consider a tracker "confirmed"

        const int sz[3]= {7, 7, 20}; 

        const float NMS_THRESHOLD;
        const float OBJ_THRESHOLD; 


        // Returns the predictions for the object locations
        std::vector<std::pair<cv::Rect2f, int>> get_predictions() const;

        // Takes the frame as input and runs the model to detect objects
        std::vector<std::pair<cv::Rect2f, int>> detect(const cv::Mat& frame);

        // Applies non maximum suppression to filter out unnecessary detections
        void NMS(const std::vector<cv::Rect2f>& boxes, const std::vector<float>& scores, const std::vector<int>& classIds, std::vector<int>& indices)const;
        
        // Processes the outputs of the model to get the object locations and the classes
        void getBoxes(const cv::Mat& label_matrix, std::vector<std::pair<cv::Rect2f, int>>& detections)const;

        // Calculates Intersection Over Union value for the bounding boxes
        static double iou(const cv::Rect2f& bbox1, const cv::Rect2f& bbox2);

        // Sigmoid function to scale values between 0 and 1
        static float sigmoid(float in);
};

// Algorithm to match detections from the model to the tracked objects
class HungarianAlgorithm {
    public:
        HungarianAlgorithm(const std::vector<std::vector<double>>& cost_matrix);
        void solve();
        const std::vector<int>& getAssignment() const;

    private:
        int numTrackers, numDetections;
        std::vector<std::vector<double>> costMatrix;
        std::vector<int> assignment;

        bool findUnassignedRow(int& row);
        bool assign(int row, int col);
};


#endif