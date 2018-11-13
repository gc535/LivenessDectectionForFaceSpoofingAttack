# include <string>
# include <vector>

#include <opencv2/opencv.hpp>



namespace QualityChecker
{

bool extractChromaticMoment(cv::Mat& src, cv::Mat& chromatic_moment)
{	
	// src image should be color image: bgr, rgb, hsv
	if(src.channels() != 3)
	{
		std::cout<<"[ERROR]: The src image for ChromaticMoment extraction is not a 3-channel image"<<std::endl; 
		return false; 
	}

	cv::Mat hsv_src;
	cv::cvtColor(src, hsv_src, CV_BGR2HSV);
	chromatic_moment = cv::Mat( cv::Size(1,15) , CV_32FC1 ); // flat matrix: 3 channels, each 5 elements

	cv::Scalar 	mean, std, skewness;
	int mat_size = hsv_src.rows * hsv_src.cols;
	// calculate mean, std seperatly on all channels
	meanStdDev(hsv_src, mean, std);
	float error[3] = {};
	float den[3] = {};
	// histgram for each channel
	int hist0[256] = {};  
	int hist1[256] = {}; 
	int hist2[256] = {};
	float minpercent[3], maxpercent[3]; 

	for(int r = 0; r < hsv_src.rows; ++r)
	{
		for(int c = 0; c < hsv_src.cols; ++c)
		{
			// update histgram
			hist0[ hsv_src.ptr<uchar>(r)[3*c] ] += 1;
			hist1[ hsv_src.ptr<uchar>(r)[3*c+1] ] += 1;
			hist2[ hsv_src.ptr<uchar>(r)[3*c+2] ] += 1;
			// pixel error
			error[0] += hsv_src.at<cv::Vec2f>(r, c)[0] - mean.val[0];
			error[1] += hsv_src.at<cv::Vec2f>(r, c)[1] - mean.val[1];
			error[2] += hsv_src.at<cv::Vec2f>(r, c)[2] - mean.val[2];
			// cube the error for skewness calculation
			skewness.val[0] += error[0]*error[0]*error[0];
			skewness.val[1] += error[1]*error[1]*error[1];
			skewness.val[2] += error[2]*error[2]*error[2];

			den[0] += error[0]*error[0];
			den[1] += error[1]*error[1];
			den[2] += error[2]*error[2];
		}
	}
	// find max and min bin pixel pencentage for all channels
	minpercent[0] = *std::min_element(std::begin(hist0), std::end(hist0)) / mat_size;
	minpercent[1] = *std::min_element(std::begin(hist1), std::end(hist1)) / mat_size;
	minpercent[2] = *std::min_element(std::begin(hist2), std::end(hist2)) / mat_size;

	maxpercent[0] = *std::max_element(std::begin(hist0), std::end(hist0)) / mat_size;
	maxpercent[1] = *std::max_element(std::begin(hist1), std::end(hist1)) / mat_size;
	maxpercent[2] = *std::max_element(std::begin(hist2), std::end(hist2)) / mat_size;

	for(int channel = 0; channel < hsv_src.channels(); ++channel)
	{
		chromatic_moment.at<float>(1, 5*channel)     = mean.val[channel];
		chromatic_moment.at<float>(1, 5*channel + 1) = std.val[channel];
		chromatic_moment.at<float>(1, 5*channel + 2) = skewness.val[channel] * sqrt(mat_size) / (den[channel] * sqrt(den[channel]));
		chromatic_moment.at<float>(1, 5*channel + 3) = minpercent[channel];
		chromatic_moment.at<float>(1, 5*channel + 4) = maxpercent[channel];
	}
	
	return true;
}	 


};
