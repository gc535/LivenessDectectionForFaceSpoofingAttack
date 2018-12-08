#ifndef UTIL_HPP
#define UTIL_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/hdf/hdf5.hpp>

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <sys/stat.h>
#include <dirent.h>
#include <errno.h>
#include <unistd.h>

cv::Mat mergeRows(const cv::Mat& A, const cv::Mat& B);
cv::Mat mergeCols(const cv::Mat& A, const cv::Mat& B);
const std::vector<cv::Mat> splitChannels(const cv::Mat& MultiChannelImage);
bool exists (const std::string& name);
void writeMatToFile(cv::Mat& m, const std::string& filename);
void saveMatToHDF5_single(cv::Mat& input, const std::string filename, const std::string type);
void saveMatToHDF5(const cv::Mat data, const cv::Mat label, const std::string filename);
int getFilelist( std::string dirPath, std::vector<std::string>& filelist);
void readFile2Mat(cv::Mat& m, const std::string file);

#endif //UTIL_HPP