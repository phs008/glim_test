//
// Created by joshua on 2024. 9. 23..
//

#pragma once
#include "odometry_estimation_base.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

/// Ceres solver 를 활용하여 Re
struct ReprojectionError {
  ReprojectionError(double _observe_X, double _observe_Y) : observe_x(_observe_X), observe_y(_observe_Y) {}
  template <typename T>
  bool operator()(const T* const camera, const T* const point, T* residuals) const {
    T p[3];
    ceres::AngleAxisRotatePoint(camera, point, p);
    p[0] += camera[3];
    p[1] += camera[4];
    p[2] += camera[5];

    T xp = p[0] / p[2];
    T yp = p[0] / p[2];

    T predicate_x = xp;
    T predicate_y = yp;
    residuals[0] = predicate_x - T(observe_x);
    residuals[1] = predicate_y - T(observe_y);
    return true;
  }
  static ceres::CostFunction* BA_CostFunction(const double x, const double y) { return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3>(new ReprojectionError(x, y)); }
  double observe_x;
  double observe_y;
};

class Odometry_estimation_imgonly : public glim::OdometryEstimationBase {
public:
  Odometry_estimation_imgonly();
  virtual ~Odometry_estimation_imgonly() override;
  virtual void insert_image(const double stamp, const cv::Mat& image) override;
  void processFrame(const cv::Mat& frame);
  void initializeMap(const cv::Mat& frame);
  void trackingFrame(const cv::Mat& frame);
  void bundleAdjustment();

private:
  cv::Ptr<cv::Feature2D> feature_detector;
  cv::Ptr<cv::DescriptorMatcher> feature_matcher;
  std::vector<cv::KeyPoint> prev_keyPoints;
  cv::Mat prev_descriptor;
  cv::Mat prev_image;
  std::vector<cv::Point3f> map_points;
  cv::Mat camera_matrix;
  std::vector<cv::Mat> camera_poses;
  bool is_initialized = false;

};
