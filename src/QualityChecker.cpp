# include <string>
# include <vector>

#include <opencv2/opencv.hpp>

namespace QualityChecker
{


bool extractDifusedComp(cv::Mat& src, cv::Mat& difused_imaged, const int max_iter)
{
	if(src.channels() != 3)
	{
		std::cout<<"[ERROR]: The src image for ChromaticMoment extraction is not a 3-channel image"<<std::endl; 
		return false; 
	}

	// iteratively remove specular component
	int iter = 0;
	while(iter < max_iter)
	{
		for(int r = 0; r < src.rows; ++r)
		{
			for(int c = 0; c < src.cols; ++c)
			{
				continue;
			}	
		}


	}


	return true;
}

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
	chromatic_moment = cv::Mat( cv::Size(15,1) , CV_32F ); // flat matrix: 3 channels, each 5 elements

	cv::Scalar 	mean, std, skewness;
	int mat_size = hsv_src.rows * hsv_src.cols;
	// calculate mean, std seperatly on all channels
	meanStdDev(hsv_src, mean, std);
	//std::cout<< mean << std::endl;
	//std::cout<< std << std::endl;
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
			//std::cout<< "bin: " << static_cast<unsigned short>(hsv_src.ptr<uchar>(r)[3*c]) << " count: " << hist0[ hsv_src.ptr<uchar>(r)[3*c] ] << std::endl;
			// update histgram
			hist0[ hsv_src.ptr<uchar>(r)[3*c] ] += 1;
			hist1[ hsv_src.ptr<uchar>(r)[3*c+1] ] += 1;
			hist2[ hsv_src.ptr<uchar>(r)[3*c+2] ] += 1;
			// pixel error
			error[0] += hsv_src.ptr<uchar>(r)[3*c] - mean.val[0];
			error[1] += hsv_src.ptr<uchar>(r)[3*c+1] - mean.val[1];
			error[2] += hsv_src.ptr<uchar>(r)[3*c+2] - mean.val[2];

			// cube the error for skewness calculation
			//std::cout<< "location: " << r<< ", " << c << " value: "  << static_cast<unsigned short>(hsv_src.ptr<uchar>(r)[3*c]) << std::endl;
			skewness.val[0] += error[0]*error[0]*error[0];
			skewness.val[1] += error[1]*error[1]*error[1];
			skewness.val[2] += error[2]*error[2]*error[2];

			den[0] += error[0]*error[0];
			den[1] += error[1]*error[1];
			den[2] += error[2]*error[2];
		}
	}
	//std::cout<< skewness << std::endl;
	// find max and min bin pixel pencentage for all channels
	//std::cout<< "min bin count: " << *std::min_element(std::begin(hist0), std::end(hist0)) << std::endl;
	minpercent[0] = *std::min_element(std::begin(hist0), std::end(hist0)) / mat_size;
	minpercent[1] = *std::min_element(std::begin(hist1), std::end(hist1)) / mat_size;
	minpercent[2] = *std::min_element(std::begin(hist2), std::end(hist2)) / mat_size;

	maxpercent[0] = static_cast<float>(*std::max_element(std::begin(hist0), std::end(hist0))) / mat_size;
	maxpercent[1] = static_cast<float>(*std::max_element(std::begin(hist1), std::end(hist1))) / mat_size;
	maxpercent[2] = static_cast<float>(*std::max_element(std::begin(hist2), std::end(hist2))) / mat_size;

	for(int channel = 0; channel < hsv_src.channels(); ++channel)
	{	
		///std::cout<< static_cast<float>(mean.val[channel]) << std::endl;
		chromatic_moment.at<float>(5*channel)     = mean.val[channel];
		chromatic_moment.at<float>(5*channel + 1) = std.val[channel];
		chromatic_moment.at<float>(5*channel + 2) = skewness.val[channel] * sqrt(mat_size) / (den[channel] * sqrt(den[channel]));
		chromatic_moment.at<float>(5*channel + 3) = minpercent[channel];
		chromatic_moment.at<float>(5*channel + 4) = maxpercent[channel];
	}
	
	return true;
}	 


bool calcBlurness(cv::Mat& src, cv::Mat score)
{
	// has to be a one channel grayscale image
	if(src.channels() == 3) 
	{
		std::cout<<"[ERROR]: The src image for blurness calculation is not a grayscale image"<<std::endl; 
		return false; 
	}

	score = cv::Mat( cv::Size(2,1) , CV_32F ); // flat matrix: 1 row, 2 cols
	
	// first metric
	cv::Mat blur, ver_blured, hor_blured;
	//cv::blur(src, ver_blured, cv::Size(9,1), cv::Point(-1,-1), cv::BORDER_DEFAULT);
	//cv::blur(src, ver_blured, cv::Size(1,9), cv::Point(-1,-1), cv::BORDER_DEFAULT);
	cv::blur(src, blur, cv::Size(5,5), cv::Point(-1,-1), cv::BORDER_DEFAULT);
	float dist_sum = 0;
	for(int r = 0; r < src.rows; ++r)
	{
		for(int = 0; c < src.cols; ++c)
		{
			dist_sum += abs(blur.at<uchar>(r, c) - src.at<uchar>(r, c)) / src.at<uchar>(r, c);
		}
	}
	score.at<float>(0) = dist_sum / (blur.rows * blur.cols)

	// second metric

}

};

int main()
{
	std::string fake = "/home/ubuntu/Desktop/d/combined/train/fake-8527.jpg";
	std::string live = "/home/ubuntu/Desktop/d/combined/train/living-6869.jpg";

	cv::Mat srcImg = cv::imread(fake, cv::IMREAD_COLOR);
	cv::Mat resizedImg;
	cv::resize(srcImg, resizedImg, cv::Size(64, 64));
	cv::Mat chromatic_moment;
	QualityChecker::extractChromaticMoment(resizedImg, chromatic_moment);
	std::cout<< chromatic_moment << std::endl;

	srcImg = cv::imread(live, cv::IMREAD_COLOR);
	cv::resize(srcImg, resizedImg, cv::Size(64, 64));
	QualityChecker::extractChromaticMoment(resizedImg, chromatic_moment);
	std::cout<< chromatic_moment << std::endl;

}