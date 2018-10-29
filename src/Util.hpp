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
void saveMatToHDF5_single(cv::Mat& input, const std::string filename, const std::string type);
void saveMatToHDF5(const cv::Mat data, const cv::Mat label, const std::string filename);
int getFilelist( std::string dirPath, std::vector<std::string>& filelist);

#endif //UTIL_HPP