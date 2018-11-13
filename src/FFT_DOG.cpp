#include <FFT_DOG.hpp>

int main(int argc, char** argv)
{	
	int resize = 96, cellsize = 16;
	double sigma1 = 1, sigma2 = 2;
	std::string train_list = "", test_list = "";
	parseArguments(argc, argv,
				   resize, cellsize, train_list, test_list);

	Data data(train_list, Data::Action::TRAIN);
	cv::Mat train_data, train_label;
	data.DataPreparation(FFTDOG, train_data, train_label, "fft_dog", resize, cellsize, sigma1, sigma2);

	return 0;
}


// batch data preparation with single_FFTDOG
void FFTDOG(cv::Mat& data, cv::Mat& label, const std::vector<std::string>& filelist, 
		    const int resize, const int cellsize, const double sigma1, const double sigma2)
{
	cv::Mat srcImg;
	cv::Mat resizedImg;

	int total_ticks = filelist.size();
	ProgressBar progressBar(total_ticks, 70, '=', '-');
	for(std::vector<std::string>::const_iterator it = filelist.begin(); it != filelist.end(); ++it)
	{
		srcImg = cv::imread(*it, cv::IMREAD_GRAYSCALE);
		cv::resize(srcImg, resizedImg, cv::Size(resize, resize));
		cv::Mat sample_dog_fft;
		sample_dog_fft = single_FFTDOG(resizedImg, sigma1, sigma2);

		// push back sample
		data.push_back(sample_dog_fft);

		/* prepare label feature vector */
		if((*it).find("fake") != std::string::npos)  
		{
			label.push_back(0);
		}
		else if((*it).find("living") != std::string::npos)  
		{
			label.push_back(1);
		}

		++progressBar;
		progressBar.display();
	}
	progressBar.done();

}

// FFT on DOG Features
cv::Mat single_FFTDOG(cv::Mat srcImg, double sigma1, double sigma2)
{
	cv::Mat XF1, XF2, DXF, output;
	int size1, size2;
	// Filter Sizes
	size1 = 2 * (int)(3*sigma1) + 3;
	size2 = 2 * (int)(3*sigma2) + 3;
	// Gaussian Filter
	cv::GaussianBlur(srcImg, XF1, cv::Size(size1, size1), sigma1, sigma1, cv::BORDER_REPLICATE);
	cv::GaussianBlur(srcImg, XF2, cv::Size(size2, size2), sigma2, sigma2, cv::BORDER_REPLICATE);
	// Difference
	DXF = XF1 - XF2;
	// Discrete Fourier Transform
	DXF.convertTo(DXF, CV_64FC1);
	cv::dft(DXF, output);
	return abs(output);
}


void parseArguments(const int argc, const char* const* argv,
					int& resize, int& cellsize, std::string& train_list, std::string& test_list)
{
	if( argc <= 2 ) {
		printHelp();
		exit( 1 );
	}
	else if( argc > 2 ) {
		// process arguments
		for( int i = 1; i < argc - 1; i++ ) {
			if( strcmp( argv[i], "-t" ) == 0 ) {
				train_list = argv[i + 1];
				i++;
			}
			else if( strcmp( argv[i], "-v" ) == 0 ){
				test_list = argv[i + 1];
				i++;

			}
			else if( strcmp( argv[i], "-r" ) == 0 ) {
				resize = atoi( argv[i + 1] );
				i++;
			}
			else if( strcmp( argv[i], "-c" ) == 0 ) {
				cellsize = atoi( argv[i + 1] );
				i++;
			}
			else {
				std::cerr << "invalid argument: \'" << argv[i] << "\'\n";
				printHelp();
				exit( 1 );
			}
		}
	}
	if (train_list == "" or test_list == ""){
		printHelp();
		exit( 1 );
	}

}

void printHelp()
{
	std::cout << "\nUsage: ./lbp [options]" << std::endl;
	std::cout << "\nOptions:" << std::endl;
	std::cout << "\t-r  <int> - Target resize size (default=96)" << std::endl;
	std::cout << "\t-c  <int> - Desired cell size for LBP extraction (default=16)" << std::endl;
	std::cout << "\t-t <string> - Path to a txt file contains a list of training data" << std::endl;
	std::cout << "\t-v <string> - Path to a txt file contains a list of testing data" << std::endl;
}