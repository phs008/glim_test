//
// Created by joshua on 2024. 9. 23..
//

#include <glim/odometry/odometry_estimation_imgonly.h>

Odometry_estimation_imgonly::Odometry_estimation_imgonly() {
  feature_detector = cv::ORB::create();
  feature_matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
  camera_matrix = (cv::Mat_<double>(3, 3) << 640, 0, 320, 0, 480, 240, 0, 0, 1);
};
void Odometry_estimation_imgonly::insert_image(const double stamp, const cv::Mat& image) {
  visualOdometry(image);
}

void Odometry_estimation_imgonly::visualOdometry(const cv::Mat& current_image) {
  std::vector<cv::KeyPoint> current_key_points;
  cv::Mat current_descriptors;
  feature_detector->detectAndCompute(current_image, cv::noArray(), current_key_points, current_descriptors);
  /// 매칭할 이미지가 있으면
  if (!prev_image.empty()) {
    std::vector<cv::DMatch> matches;
    feature_matcher->match(prev_descriptor, current_descriptors, matches);
    std::vector<cv::DMatch> good_matches;
    for (const auto& match : good_matches) {
      if (match.distance < 50) {
        good_matches.push_back(match);
      }
    }
    std::vector<cv::Point2f> prev_points, current_points;
    for (const auto& match : good_matches) {
      prev_points.push_back(prev_keyPoints[match.queryIdx].pt);
      current_points.push_back(current_key_points[match.trainIdx].pt);
    }
    /// Essential matrix 계산
    cv::Mat E, mask;
    E = cv::findEssentialMat(prev_points, current_points, camera_matrix, cv::RANSAC, 0.99, 1.0, mask);
    /// 카메라 Pose estimations
    cv::Mat R, t;
    cv::recoverPose(E, prev_points, current_points, camera_matrix, R, t, mask);

    /// 삼각측량 으로 3D Point 생성
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
      prev_points,
      current_points,
      points4D);

    // 동차 좌표계 -> 카테시안 좌표계
    for (int i = 0; i < points4D.cols; ++i) {
      if (mask.at<uchar>(i)) {
        cv::Mat x = points4D.col(i);
        x /= x.at<float>(3, 0);
        cv::Point3f p(x.at<float>(0, 0), x.at<float>(1, 0), x.at<float>(2, 0));
        map_points.push_back(p);
      }
    }

    // 카메라 pose 를 일정 frame 동안 저장해놨다가.
    cv::Mat current_pose;
    cv::hconcat(R, t, current_pose);
    camera_poses.push_back(current_pose);

    // 10 frame 에 한번씩 BA 수행.
    if (camera_poses.size() % 10 == 0) {
      bundleAdjustment();
    }

    for (const auto& point : map_points) {
      cv::Point2f pt;
      pt.x = point.x * camera_matrix.at<double>(0, 0) / point.z + camera_matrix.at<double>(0, 2);
      pt.y = point.y * camera_matrix.at<double>(1, 1) / point.z + camera_matrix.at<double>(1, 2);
      cv::circle(current_image, pt, 3, cv::Scalar(0, 255, 0), -1);
    }

    cv::imshow("Mono SLAM", current_image);
  }
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
  ceres::Solve(options , &problem , &summary);
}

