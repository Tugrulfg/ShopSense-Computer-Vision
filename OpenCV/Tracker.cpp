#include "Tracker.hpp"
#include <algorithm>
#include <limits>

// Class for storing tracked objects
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

    // Keep the enterance direction of the object
    if(state.at<float>(1)-state.at<float>(3)/2 <= 112.0)       // From Top
        enter_dir = 1;
    else                                // From Bottom
        enter_dir = 0;  

    kf.statePost = state;

    classes.push_back(class_id);
}

// Predicts the next state of the object based on the current state
void KalmanFilter::predict() {
    state = kf.predict();

    age++;
    time_since_update++;
}

// Updates the state of the object based on the measurement
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

// Returns the current location of the object
cv::Rect2f KalmanFilter::get_bbox() const {
    float cx = state.at<float>(0);
    float cy = state.at<float>(1);
    float w = state.at<float>(2);
    float h = state.at<float>(3);
    return cv::Rect2f(state.at<float>(0) - state.at<float>(2) / 2, state.at<float>(1) - state.at<float>(3) / 2, state.at<float>(2), state.at<float>(3));
}

// Returns the class of the object
int KalmanFilter::get_class_id() const {
    return class_id;
}

// Returns the time since the object was last updated
int KalmanFilter::get_time_since_update() const {
    return time_since_update;
}

// Increments the time since the object was last updated
void KalmanFilter::increment_time_since_update() {
    time_since_update++;
}

// Updates the class of the object
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

// Returns the number of times the object has been tracked
int KalmanFilter::get_hits() const {
    return hits;
}

// Returns the exit direction of the object: 0-->Bottom, 1-->Top
int KalmanFilter::exit_dir()const{
    float vy = state.at<float>(5); // Velocity in y direction
    if(enter_dir == 0 && vy > 0 || enter_dir == 1 && vy < 0) 
        return -1;
    return (vy < 0) ? 1 : 0; // 1 for top exit, 0 for bottom exit
}


MyTracker::MyTracker(const std::string model_path, const float nms_thres, const float obj_thres): NMS_THRESHOLD(nms_thres), OBJ_THRESHOLD(obj_thres){
    model = cv::dnn::readNetFromONNX(model_path);
}

MyTracker::~MyTracker(){

}

// Takes the input frame and updates the tracked objects
std::vector<std::pair<int, int>> MyTracker::update(const cv::Mat& frame) {

    std::vector<std::pair<cv::Rect2f, int>> detections = detect(frame);
    
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

// Draws the detections to the frame
void MyTracker::draw(cv::Mat& frame)const{
    for(const std::pair<cv::Rect2f, int>& prediction : get_predictions()){
        cv::rectangle(frame, prediction.first, cv::Scalar(0, 255, 255), 2); // Different color for predictions
        cv::putText(frame, std::to_string(prediction.second), prediction.first.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 2);
    }
}

// Returns the predicted locations of the objects
std::vector<std::pair<cv::Rect2f, int>> MyTracker::get_predictions() const {
    std::vector<std::pair<cv::Rect2f, int>> predictions;
    for(const auto& tracker : trackers){
        if(tracker.get_hits() >= min_hits)  // Only return confirmed tracks
            predictions.emplace_back(tracker.get_bbox(), tracker.get_class_id());
    }
    return predictions;
}

// Takes the input frame and updates the tracked objects
std::vector<std::pair<cv::Rect2f, int>> MyTracker::detect(const cv::Mat& frame){
    cv::Mat input, output;
    std::vector<std::pair<cv::Rect2f, int>> detections;
    cv::dnn::blobFromImage(frame, input, 1.0/255.0, cv::Size(224., 224.), cv::Scalar(), true, false);
    model.setInput(input);

    output = model.forward();
    cv::Mat newmat(3, sz, output.type(), output.ptr<float>(0));
        
    getBoxes(newmat, detections);
    return detections;
}

// Applies non maximum suppression to filter out unnecessary detections
void MyTracker::NMS(const std::vector<cv::Rect2f>& boxes, const std::vector<float>& scores, const std::vector<int>& classIds, std::vector<int>& indices) const{
    std::vector<int> order(boxes.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&scores](int a, int b) { return scores[a] > scores[b]; });

    std::vector<bool> suppressed(boxes.size(), false);

    for (size_t i = 0; i < order.size(); ++i) {
        int idx = order[i];
        if (suppressed[idx]) continue;
        indices.push_back(idx);

        for (size_t j = i + 1; j < order.size(); ++j) {
            int nextIdx = order[j];
            if (classIds[idx] != classIds[nextIdx]) continue; // Skip different classes
            if (iou(boxes[idx], boxes[nextIdx]) > NMS_THRESHOLD) {
                suppressed[nextIdx] = true;
            }
        }
    }
}

// Processes the outputs of the model to get the object locations and the classes
void MyTracker::getBoxes(const cv::Mat& label_matrix, std::vector<std::pair<cv::Rect2f, int>>& detections) const{
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect2f> boxes;
    for (int i = 0; i < 7; ++i) {
        for(int j=0; j<7; j++){
            if(label_matrix.at<float>(i, j, 10) > label_matrix.at<float>(i, j, 15)){
                float confidence = label_matrix.at<float>(i, j, 10);
                confidence = sigmoid(confidence);
                if (confidence > OBJ_THRESHOLD) {
                    int centerX = static_cast<int>((label_matrix.at<float>(i, j, 11) + j) * (224./7));
                    int centerY = static_cast<int>((label_matrix.at<float>(i, j, 12) + i) * (224./7));
                    int width = static_cast<int>(label_matrix.at<float>(i, j, 13) * 224.);
                    int height = static_cast<int>(label_matrix.at<float>(i, j, 14) * 224.);
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
                    int centerX = static_cast<int>((label_matrix.at<float>(i, j, 16) + j) * (224./7));
                    int centerY = static_cast<int>((label_matrix.at<float>(i, j, 17) + i) * (224./7));
                    int width = static_cast<int>(label_matrix.at<float>(i, j, 18) * 224.);
                    int height = static_cast<int>(label_matrix.at<float>(i, j, 19) * 224.);
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
    NMS(boxes, confidences, classIds, indices);

    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        detections.push_back({boxes[idx], classIds[idx]});
    }
}

// Calculates the Intersection Over Union value between two bounding boxes
double MyTracker::iou(const cv::Rect2f& bbox1, const cv::Rect2f& bbox2){
    float intersection = std::max(0.0f, std::min(bbox1.x + bbox1.width, bbox2.x + bbox2.width) - std::max(bbox1.x, bbox2.x)) * std::max(0.0f, std::min(bbox1.y + bbox1.height, bbox2.y + bbox2.height) - std::max(bbox1.y, bbox2.y));

    return intersection / (bbox1.width * bbox1.height + bbox2.width * bbox2.height - intersection + 1e-7);
}

// Sigmoid function to scale values between 0 and 1
float MyTracker::sigmoid(float in) {
    return 1.0 / (1.0 + std::exp(-in));
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
