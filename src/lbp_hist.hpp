#ifndef __LBP_HIST_H__
#define __LBP_HIST_H__
#include <opencv2/opencv.hpp>

void getFaceLBPHist(cv::Mat &face, std::vector<double> &hist, const int cellsize);
#endif