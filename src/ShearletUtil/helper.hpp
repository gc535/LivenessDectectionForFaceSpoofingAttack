#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

void gen_x_y_cordinates(cv::Mat& x1n, cv::Mat& y1n, cv::Mat& x2n, cv::Mat& y2n, cv::Mat& D, int n);

std::vector<double> linespace(double start, double end, int num_sample);

cv::Mat avg_pol(int L, const cv::Mat& x1, const cv::Mat& y1, const cv::Mat& x2, const cv::Mat& y2);

cv::Mat rec_from_pol(cv::Mat& l, int n, cv::Mat& x1, cv::Mat& y1, cv::Mat& x2, cv::Mat& y2, cv::Mat& D);
