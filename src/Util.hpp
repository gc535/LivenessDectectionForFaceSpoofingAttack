#ifndef UTIL_HPP
#define UTIL_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

cv::Mat mergeRows(const cv::Mat& A, const cv::Mat& B);
cv::Mat mergeCols(const cv::Mat& A, const cv::Mat& B);
const std::vector<cv::Mat> splitChannels(const cv::Mat& MultiChannelImage);
bool exists (const std::string& name);
void writeMatToFile(cv::Mat& m, const std::string& filename);

#endif //UTIL_HPP