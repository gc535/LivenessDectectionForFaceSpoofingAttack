#pragma once 

#include <brisque.hpp>
#include <Data.hpp>
#include <Util.hpp>
#include <ProgressBar.hpp>

void extractQualityFeature(cv::Mat& data, cv::Mat& label, const std::vector<std::string>& filelist);

void parseArguments(const int argc, const char* const* argv,
                    int& resize, int& cellsize, std::string& train_list, std::string& test_list);

void printHelp();