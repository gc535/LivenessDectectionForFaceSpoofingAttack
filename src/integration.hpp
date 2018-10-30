#ifndef INTEGRATION_HPP
#define INTEGRATION_HPP

#include <opencv2/opencv.hpp>

#include <stdlib.h>
#include <string>

#define CPU 0
#define OpenCL 1
#define OpenCLF16 2
#define VPU 3

int dog_lbp_extraction(cv::Mat& img, cv::Mat& feature_vector, const int resize, const int cell_size);
int ofm_extraction(cv::Mat& img, cv::Mat& feature_vector, const int cell_size);
int gray_lbp_extraction(cv::Mat& img, cv::Mat& feature_vector, const int cell_size);

void ParseArgument(const int& argc, const char* const* argv, 
				   std::string& model_1, std::string& checkpoint_1, 
				   std::string& model_2, std::string& checkpoint_2, 
				   std::string& model_3, std::string& checkpoint_3,
				   int& resize, int& cell_size,
				   std::string& dog_lbp_data_path, std::string& ofm_data_path, std::string& gray_img_path);
void printHelp();


#endif //INTEGRATION_HPP