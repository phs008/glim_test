//
// Created by joshua on 2024. 9. 23..
//

#include <boost/math/special_functions/math_fwd.hpp>
#include <glim/odometry/odometry_estimation_imgonly.h>
#include <gtsam/inference/BayesNet-inst.h>

Odometry_estimation_imgonly::Odometry_estimation_imgonly() {
  feature_detector = cv::ORB::create();
  feature_matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
  camera_matrix = (cv::Mat_<double>(3, 3) << 640, 0, 320, 0, 480, 240, 0, 0, 1);
};
void Odometry_estimation_imgonly::insert_image(const double stamp, const cv::Mat& image) {
  processFrame(image);
}

void Odometry_estimation_imgonly::processFrame(const cv::Mat& frame) {
  if (!is_initialized) {
    if (prev_image.empty()) {
      frame.copyTo(prev_image);
      feature_detector->detectAndCompute(frame, cv::noArray(), prev_keyPoints, prev_descriptor);
    } else {
      initializeMap(frame);
    }
  } else {
    trackingFrame(frame);
    if (camera_poses.size() % 10 == 0) bundleAdjustment();
  }
  // 매칭할 이미지 최신화
  prev_image = current_image;
}
void Odometry_estimation_imgonly::initializeMap(const cv::Mat& frame) {
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  feature_detector->detectAndCompute(frame, cv::noArray(), keypoints, descriptors);
  std::vector<cv::DMatch> matches;
  feature_matcher->match(prev_descriptor, descriptors, matches);
  std::vector<cv::Point2d> points1f, points2f;
  for (const auto& match : matches) {
    if (match.distance < 50) {
      points1f.push_back(prev_keyPoints[match.queryIdx].pt);
      points2f.push_back(keypoints[match.trainIdx].pt);
    }
  }
  cv::Mat E, R, t;
  E = cv::findEssentialMat(points1f, points2f, camera_matrix);
  cv::recoverPose(points1f, points2f, camera_matrix, R, t, cv::noArray());

  cv::Mat points4D;
  cv::triangulatePoints(
    cv::Mat::eye(3, 4, CV_64F),
    cv::Matx34d(
      R.at<double>(0, 0),
      R.at<double>(0, 1),
      R.at<double>(0, 2),
      t.at<double>(0),
      R.at<double>(1, 0),
      R.at<double>(1, 1),
      R.at<double>(1, 2),
      t.at<double>(1),
      R.at<double>(2, 0),
      R.at<double>(2, 1),
      R.at<double>(2, 2),
      t.at<double>(2)),
    points1f,
    points2f,
    points4D);

  for (int i = 0; i < points4D.cols; ++i) {
    cv::Mat x = points4D.col(i);
    x /= x.at<float>(3, 0);
    cv::Point3f p(x.at<float>(0, 0), x.at<float>(1, 0), x.at<float>(2, 0));
    map_points.push_back(p);
  }

  cv::Mat current_pose;
  cv::hconcat(R, t, current_pose);
  camera_poses.push_back(current_pose);

  is_initialized = true;
  frame.copyTo(prev_image);
  prev_keyPoints = keypoints;
  prev_descriptor = descriptors;
}
void Odometry_estimation_imgonly::trackingFrame(const cv::Mat& frame) {
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  feature_detector->detectAndCompute(frame, cv::noArray(), keypoints, descriptors);
  std::vector<cv::DMatch> matches;
  feature_matcher->match(prev_descriptor, descriptors, matches);
  std::vector<cv::Point3f> object_points;
  std::vector<cv::Point2d> image_points;
  for (const auto& match : matches) {
    if (match.distance < 50) {
      object_points.push_back(map_points[match.queryIdx]);
      image_points.push_back(keypoints[match.trainIdx].pt);
    }
  }

  cv::Mat rvec, tvec;
  cv::solvePnP(object_points, image_points, camera_matrix, cv::noArray(), rvec, tvec);

  cv::Mat R;
  cv::Rodrigues(rvec, R);
  cv::Mat pose = cv::Mat::eye(4, 4, CV_32F);
  R.copyTo(pose(cv::Rect(0, 0, 3, 3)));
  tvec.copyTo(pose(cv::Rect(3, 0, 1, 3)));

  camera_poses.push_back(pose);

  frame.copyTo(prev_image);
  prev_keyPoints = keypoints;
  prev_descriptor = descriptors;
}

void Odometry_estimation_imgonly::bundleAdjustment() {
  ceres::Problem problem;
  for (size_t i = 0; i < camera_poses.size(); ++i) {
    double* camera = camera_poses[i].ptr<double>();
    for (size_t j = 0; i < map_points.size(); ++j) {
      double* point = (double*)&map_points[j];

      cv::Mat pt_3d = (cv::Mat_<double>(4, 1) << point[0], point[1], point[2], 1.0);
      cv::Mat pt_2d = camera_matrix * camera_poses[i] * pt_3d;
      cv::Point2f p(pt_2d.at<double>(0) / pt_2d.at<double>(2), pt_2d.at<double>(0) / pt_2d.at<double>(2));

      ceres::CostFunction* cost_function = ReprojectionError::BA_CostFunction(p.x, p.y);
      problem.AddResidualBlock(cost_function, nullptr, camera, point);
    }
  }
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = false;
  options.max_num_iterations = 100;
  options.num_threads = 8;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
}
