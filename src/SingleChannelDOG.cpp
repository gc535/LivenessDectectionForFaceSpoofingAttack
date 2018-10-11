#include <SingleChannelDOG.hpp>


void SingalChannelImageDoG(const cv::Mat& input_image, const std::vector<cv::Vec2d>& sigmas, std::vector<cv::Mat>& output_dog_images)
{
	try
	{
		for(std::vector<cv::Vec2d>::const_iterator sigma = sigmas.begin(); sigma != sigmas.end(); ++sigma)
		{
			cv::Mat output_dog_image;// = cv::Mat::zeros(input_image.rows, input_image.cols, CV_32FC1);
			if(SingalChannelImageDoG(input_image, *sigma, output_dog_image) != SUCCESS)
			{
				throw std::runtime_error("[Error]: Cannot generate DOG on the imput image");
			}
			output_dog_images.push_back(output_dog_image);
		}
	}
	catch(cv::Exception& e) 
	{
		std::cout << e.msg;
	}
}


int SingalChannelImageDoG(const cv::Mat& input_image, const cv::Vec2d& sigma, cv::Mat& output_dog_image) 
{
  if (input_image.empty()) {
    std::cout << "[Error]: The input image is empty"<<std::endl;
    return FAILURE;
  }
  if (input_image.channels() != 1) {
    std::cout << "[Error]: The input image is not singal channal image"<<std::endl;
    return FAILURE;
  }
  cv::Mat gaussian_image1, gaussian_image2;
  int size1, size2;
  // Filter Sizes
  size1 = 2 * (int)(3 * sigma[0]) + 3;
  size2 = 2 * (int)(3 * sigma[1]) + 3;
  //std::cout<<"input image row: "<<input_image.rows<<"col: "<<input_image.cols<<std::endl;
  // Gaussian Filter
  cv::GaussianBlur(input_image, gaussian_image1, cv::Size(size1, size1), sigma[0], sigma[0], cv::BORDER_REPLICATE);
  cv::GaussianBlur(input_image, gaussian_image2, cv::Size(size2, size2), sigma[1], sigma[1], cv::BORDER_REPLICATE);
  //std::cout<<"gaussian row: "<<gaussian_image1.rows<<"col: "<<gaussian_image1.cols<<std::endl;
  // Difference
  output_dog_image.create(gaussian_image1.size(), CV_8UC1);
  for (int i = 0; i < output_dog_image.rows; i++) {
    for (int j = 0; j < output_dog_image.cols; j++) {
      output_dog_image.at<unsigned char>(i, j) = (unsigned char)abs((gaussian_image1.at<unsigned char>(i, j) - gaussian_image2.at<unsigned char>(i, j)));
    }
  }
  //difference_image.copyTo(output_dog_image);
  return SUCCESS;
}

/*
int SingalChannelImageDoG(const cv::Mat& input_image, const cv::Vec2d& sigma, cv::Mat& output_dog_image) 
{
  if (input_image.empty()) {
    std::cout << "[Error]: The input image is empty"<<std::endl;
    return FAILURE;
  }
  if (input_image.channels() != 1) {
    std::cout << "[Error]: The input image is not singal channal image"<<std::endl;
    return FAILURE;
  }
  cv::Mat gaussian_image1, gaussian_image2;
  int size1, size2;
  // Filter Sizes
  size1 = 2 * (int)(3 * sigma[0]) + 3;
  size2 = 2 * (int)(3 * sigma[1]) + 3;
  // Gaussian Filter
  cv::GaussianBlur(input_image, gaussian_image1, cv::Size(size1, size1), sigma[0], sigma[0], cv::BORDER_REPLICATE);
  cv::GaussianBlur(input_image, gaussian_image2, cv::Size(size2, size2), sigma[1], sigma[1], cv::BORDER_REPLICATE);
  // Difference
  cv::Mat difference_image = cv::Mat::zeros(gaussian_image1.rows, gaussian_image1.cols, CV_32FC1);
  for (int i = 0; i < difference_image.rows; i++) {
    for (int j = 0; j < difference_image.cols; j++) {
      difference_image.at<float>(i, j) = (float)abs((gaussian_image1.at<float>(i, j) - gaussian_image2.at<float>(i, j)));
    }
  }
  difference_image.copyTo(output_dog_image);
  return SUCCESS;
}
*/