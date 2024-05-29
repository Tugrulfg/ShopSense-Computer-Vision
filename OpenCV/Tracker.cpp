#include "Tracker.hpp"
#include <algorithm>
#include <limits>

KalmanFilter::KalmanFilter(const cv::Rect2f& init_bbox, int class_id):age(0), hits(0), time_since_update(0) {
    // Initialize Kalman filter parameters
    kf = cv::KalmanFilter(7, 4, 0);
    state = cv::Mat::zeros(7, 1, CV_32F);
    meas = cv::Mat::zeros(4, 1, CV_32F);

    // Transition Matrix
    kf.transitionMatrix = (cv::Mat_<float>(7, 7) << 
        1, 0, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 1, 0,
        0, 0, 1, 0, 0, 0, 1,
        0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 1);

    kf.measurementMatrix = (cv::Mat_<float>(4, 7) <<
        1, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0);

    kf.processNoiseCov = cv::Mat::eye(7, 7, CV_32F) * 1e-2;
    kf.measurementNoiseCov = cv::Mat::eye(4, 4, CV_32F) * 1e-1;

    // Initial state
    state.at<float>(0) = init_bbox.x + init_bbox.width / 2;
    state.at<float>(1) = init_bbox.y + init_bbox.height / 2;
    state.at<float>(2) = init_bbox.width;
    state.at<float>(3) = init_bbox.height;

    kf.statePost = state;

    classes.push_back(class_id);
}

void KalmanFilter::predict() {
    state = kf.predict();

    age++;
    time_since_update++;
}

void KalmanFilter::update(const cv::Rect2f& bbox, int class_id) {
    meas.at<float>(0) = bbox.x + bbox.width / 2;
    meas.at<float>(1) = bbox.y + bbox.height / 2;
    meas.at<float>(2) = bbox.width;
    meas.at<float>(3) = bbox.height;
    kf.correct(meas);
    
    update_class(class_id);

    hits++;
    time_since_update = 0;
}

cv::Rect2f KalmanFilter::get_bbox() const {
    float cx = state.at<float>(0);
    float cy = state.at<float>(1);
    float w = state.at<float>(2);
    float h = state.at<float>(3);
    return cv::Rect2f(state.at<float>(0) - state.at<float>(2) / 2, state.at<float>(1) - state.at<float>(3) / 2, state.at<float>(2), state.at<float>(3));
}

int KalmanFilter::get_class_id() const {
    return class_id;
}

int KalmanFilter::get_time_since_update() const {
    return time_since_update;
}

void KalmanFilter::increment_time_since_update() {
    time_since_update++;
}

void KalmanFilter::update_class(int new_class_id){
    // Add new class_id to history
    classes.push_back(new_class_id);
    if (classes.size() > 30) {
        classes.pop_front();
    }

    // Find the most frequent class_id in history
    std::unordered_map<int, int> class_id_count;
    for (int id : classes) {
        class_id_count[id]++;
    }

    int most_frequent_id = new_class_id;
    int max_count = 0;
    for (const auto& pair : class_id_count) {
        if (pair.second > max_count) {
            most_frequent_id = pair.first;
            max_count = pair.second;
        }
    }

    class_id = most_frequent_id;
}

int KalmanFilter::get_hits() const {
    return hits;
}

// 0-->Bottom, 1-->Top
int KalmanFilter::exit_dir()const{
    if(state.at<float>(1)-state.at<float>(3)/2 <= 112.0)
        return 1;
    return 0;
}


MyTracker::MyTracker(){

}

MyTracker::~MyTracker(){

}

double MyTracker::iou(const cv::Rect2f& bbox1, const cv::Rect2f& bbox2) const{
    float intersection = std::max(0.0f, std::min(bbox1.x + bbox1.width, bbox2.x + bbox2.width) - std::max(bbox1.x, bbox2.x)) * std::max(0.0f, std::min(bbox1.y + bbox1.height, bbox2.y + bbox2.height) - std::max(bbox1.y, bbox2.y));

    return intersection / (bbox1.width * bbox1.height + bbox2.width * bbox2.height - intersection + 1e-7);
}

std::vector<std::pair<int, int>> MyTracker::update(const std::vector<std::pair<cv::Rect2f, int>>& detections) {
    // Predict new locations for all trackers
    for (auto& tracker : trackers) {
        tracker.predict();
    }

    int num_trackers = trackers.size();
    int num_detections = detections.size();

    // Create cost matrix
    std::vector<std::vector<double>> costMatrix(num_trackers, std::vector<double>(num_detections, 1.0));

    for (int i = 0; i < num_trackers; ++i) {
        for (int j = 0; j < num_detections; ++j) {
            costMatrix[i][j] = 1.0 - iou(trackers[i].get_bbox(), detections[j].first); // 1 - IoU as cost
        }
    }


    std::vector<bool> assigned_detections(num_detections, false);
    if(num_trackers != 0 && num_detections != 0){
    // Solve assignment problem
        HungarianAlgorithm hungarian(costMatrix);
        hungarian.solve();
        const std::vector<int>& assignment = hungarian.getAssignment();

        // Update trackers with assigned detections
        for (int i = 0; i < num_trackers; ++i) {
            if (assignment[i] != -1) {
                trackers[i].update(detections[assignment[i]].first, detections[assignment[i]].second);
                assigned_detections[assignment[i]] = true;
            } else {
                trackers[i].increment_time_since_update();
            }
        }
    }
    else if(num_trackers != 0){
        for(int i=0; i<num_trackers; ++i)
            trackers[i].increment_time_since_update();
    }

    // Add new trackers for unassigned detections
    for (int j = 0; j < num_detections; ++j) {
        if (!assigned_detections[j]) {
            trackers.emplace_back(detections[j].first, detections[j].second);
        }
    }
    
    // Remove trackers that have not been updated for a while
    std::vector<std::pair<int, int>> results;
    trackers.erase(std::remove_if(trackers.begin(), trackers.end(),
                                   [this, &results](const KalmanFilter& tracker) {
                                        if(tracker.get_time_since_update() > max_age){
                                            if(tracker.get_hits() >= min_hits)
                                                results.push_back({tracker.exit_dir(), tracker.get_class_id()});
                                            return true;
                                        }
                                        return false;
                                   }),
                    trackers.end());
    return results;
}

std::vector<std::pair<cv::Rect2f, int>> MyTracker::get_tracks() const {
    std::vector<std::pair<cv::Rect2f, int>> tracks;
    for(const auto& tracker : trackers){
        if (tracker.get_hits() >= min_hits)  // Only return confirmed tracks
            tracks.emplace_back(tracker.get_bbox(), tracker.get_class_id());
    }
    return tracks;
}

std::vector<std::pair<cv::Rect2f, int>> MyTracker::get_predictions() const {
    std::vector<std::pair<cv::Rect2f, int>> predictions;
    for(const auto& tracker : trackers){
        if(tracker.get_hits() >= min_hits)  // Only return confirmed tracks
            predictions.emplace_back(tracker.get_bbox(), tracker.get_class_id());
    }
    return predictions;
}

void MyTracker::draw(cv::Mat& frame)const{
    for(const std::pair<cv::Rect2f, int>& prediction : get_predictions()){
        cv::rectangle(frame, prediction.first, cv::Scalar(0, 255, 255), 2); // Different color for predictions
        cv::putText(frame, std::to_string(prediction.second), prediction.first.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 2);
    }
}



HungarianAlgorithm::HungarianAlgorithm(const std::vector<std::vector<double>>& cost_matrix)
    : costMatrix(cost_matrix), numTrackers(cost_matrix.size()), numDetections(cost_matrix[0].size()) {}

bool HungarianAlgorithm::findUnassignedRow(int& row) {
    for (row = 0; row < numTrackers; ++row) {
        if (assignment[row] == -1) {
            return true;
        }
    }
    return false;
}

bool HungarianAlgorithm::assign(int row, int col) {
    for (int j = 0; j < numDetections; ++j) {
        if (j != col && assignment[row] == j) {
            assignment[row] = -1;
            return false;
        }
    }
    assignment[row] = col;
    return true;
}

void HungarianAlgorithm::solve() {
    assignment.assign(numTrackers, -1);
    std::vector<bool> rowCovered(numTrackers, false);
    std::vector<bool> colCovered(numDetections, false);
    int pathRow, pathCol;

    for (int i = 0; i < numTrackers; ++i) {
        double minVal = std::numeric_limits<double>::max();
        for (int j = 0; j < numDetections; ++j) {
            if (!colCovered[j] && costMatrix[i][j] < minVal) {
                minVal = costMatrix[i][j];
                pathCol = j;
            }
        }
        assignment[i] = pathCol;
        colCovered[pathCol] = true;
    }
    for (int i = 0; i < numTrackers; ++i) {
        if (assignment[i] == -1) {
            while (true) {
                if (!findUnassignedRow(pathRow)) {
                    break;
                }
                std::fill(rowCovered.begin(), rowCovered.end(), false);
                std::fill(colCovered.begin(), colCovered.end(), false);
                if (assign(pathRow, assignment[pathRow])) {
                    break;
                }
            }
        }
    }
}

const std::vector<int>& HungarianAlgorithm::getAssignment() const{
    return assignment;
}
