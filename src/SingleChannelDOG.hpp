#ifndef SINGLE_CHANNELDOG_HPP
#define SINGLE_CHANNELDOG_HPP

#include <vector>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>

#define SUCCESS 1
#define FAILURE 0

void SingalChannelImageDoG(const cv::Mat& input_image, const std::vector<cv::Vec2d>& sigmas, std::vector<cv::Mat>& output_dog_images);
 int SingalChannelImageDoG(const cv::Mat& input_image, const             cv::Vec2d & sigmas,             cv::Mat & output_dog_image );

#endif