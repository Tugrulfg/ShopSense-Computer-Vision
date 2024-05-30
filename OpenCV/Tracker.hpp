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

class KalmanFilter {
    public:
        KalmanFilter(const cv::Rect2f& init_bbox, int class_id);
        void predict();
        void update(const cv::Rect2f& bbox, int class_id);
        cv::Rect2f get_bbox() const;
        int get_class_id() const;
        int get_time_since_update() const;
        void increment_time_since_update();
        void update_class(int new_class);
        int get_hits() const;
        int exit_dir()const;        // 0-->Bottom, 1-->Top

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

class MyTracker {
    public:
        MyTracker();
        ~MyTracker();
        std::vector<std::pair<int, int>> update(const std::vector<std::pair<cv::Rect2f, int>>& detections);
        std::vector<std::pair<cv::Rect2f, int>> get_tracks() const;
        std::vector<std::pair<cv::Rect2f, int>> get_predictions() const;
        void draw(cv::Mat& frame)const;

    private:
        std::vector<KalmanFilter> trackers;
        int max_age = 10; // Maximum number of frames to keep a tracker without updates
        int min_hits = 5; // Minimum number of hits to consider a tracker "confirmed"

        double iou(const cv::Rect2f& bbox1, const cv::Rect2f& bbox2)const;
};

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