#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

#include <lbpExtraction.hpp>
#include <Data.hpp>
#include <ProgressBar.hpp>
#include <Util.hpp>
#include <SingleChannelDOG.hpp>
#include <LBP.hpp>

#include <stdlib.h>
#include <vector>
#include <string>
#include <iostream>
#include <errno.h>

int main(int argc, char** argv)
{
	int resize = 96;
	int cellsize = 16;
	Action action = TRAIN;
	std::string train_list = "", test_list = "";
	parseArguments(argc, argv,
				   resize, cellsize, train_list, test_list, action);

	std::cout<<"[Note]: Starting data preparation phase..."<<std::endl;
	// prepare train data
	Data data(train_list, Data::Action::TRAIN);
	cv::Mat train_data, train_label;
	if(action != TEST)
	{
		data.DataPreparation(DOG_LBP, train_data, train_label, "dog_lbp", resize, cellsize);
	}
	
	// prepare test data
	data.update(test_list, Data::Action::TEST);
	cv::Mat test_data, test_label;
	data.DataPreparation(DOG_LBP, test_data, test_label, "dog_lbp", resize, cellsize);


	std::cout<<"[Note]: Starting model traning phase..."<<std::endl;
	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
	// training 
	if(!exists("liveness_svm.xml"))
	{
		svm->setType(cv::ml::SVM::Types::C_SVC);
	  	svm->setKernel(cv::ml::SVM::KernelTypes::LINEAR);
	  	svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER,100,1e-6));
	  	std::cout << "[Note]: Training SVM......" << std::endl;
	  	try 
	  	{
			//svm = StatModel::train<SVM>(train_feature, ROW_SAMPLE, train_label, params);//train SVM
			svm->train(train_data, cv::ml::SampleTypes::ROW_SAMPLE, train_label);
	  		std::cout << "[Note]: Training finished......" << std::endl;
	  		svm->save("liveness_svm.xml");
	  	} 
	  	catch (cv::Exception& e) 
	  	{
			std::cout << e.msg;
	  	}
	}
	else
	{
		std::cout << "[Note]: Pre-trained SVM model found in the current directory, load it from file......" << std::endl;
		svm = cv::Algorithm::load<cv::ml::SVM>("liveness_svm.xml");
	}
  	std::cout<<"[Note]: Model traning phase complete!"<<std::endl;


	std::cout<<"[Note]: Starting model testing phase..."<<std::endl;
  	std::cout<<"[Note]: Test set size: "<<test_data.rows<<" samples"<< std::endl;
  	test(test_data, test_label, svm);

    return 0;
}


void test(cv::Mat test_data, cv::Mat test_label, cv::Ptr<cv::ml::SVM> svm)
{
	CV_Assert(test_data.rows == test_label.rows && test_data.rows>0);

	std::cout << "[Note]: Start Testing......" << std::endl;
	int count = 0;
	int correct = 0;
	ProgressBar progressBar(test_data.rows, 70, '=', '-');
	for(int r = 0; r < test_data.rows; ++r)
	{
		int response = svm->predict(test_data.row(r));
		//std::cout<<"response: "<< response << ", expected: " <<  test_label.at<int>(r, 0) << std::endl;
		if (response == test_label.at<int>(r, 0)) correct++;
		count++;
		++progressBar;
		progressBar.display();
	}
	progressBar.done();
	std::cout<<"[Note]: Current model accuracy is: " << float(correct)/count*100 << "%" << std::endl;
}


void DOG_LBP(cv::Mat& data, cv::Mat& label, const std::vector<std::string>& filelist, int resize, int cellsize )
{
	cv::Mat srcImg;
	cv::Mat resizedImg;

	LBP lbp_cell(cv::Size(cellsize, cellsize));    // setup the lbp extractor
	LBP lbp_full(cv::Size(resize, resize));		   // setup the lbp extractor

	int total_ticks = filelist.size();
	ProgressBar progressBar(total_ticks, 70, '=', '-');
	for(std::vector<std::string>::const_iterator it = filelist.begin(); it != filelist.end(); ++it)
	{
		srcImg = cv::imread(*it, cv::IMREAD_COLOR);
		cv::resize(srcImg, resizedImg, cv::Size(resize, resize));
		

		/* prepare feature vector for every sample*/
		std::vector<cv::Mat> dog_channels;
		const std::vector<cv::Vec2d> vector_sigmas = { cv::Vec2d(0.5, 1), cv::Vec2d(1, 2), cv::Vec2d(0.5,2)};
		
		
		// hsv
		cv::Mat hsv_image;
		cv::cvtColor(resizedImg, hsv_image, cv::COLOR_RGB2HSV);
		const std::vector<cv::Mat> hsv_channels = splitChannels(hsv_image);
		for(std::vector<cv::Mat>::const_iterator channel = hsv_channels.begin(); channel != hsv_channels.end(); ++channel)
		{
			SingalChannelImageDoG(*channel, vector_sigmas, dog_channels);
		}
		
		
		// ycbcr
		cv::Mat ycbcr_image;
		cv::cvtColor(resizedImg, ycbcr_image, cv::COLOR_RGB2YCrCb);
		const std::vector<cv::Mat> ycbcr_channels = splitChannels(ycbcr_image);
		for(std::vector<cv::Mat>::const_iterator channel = ycbcr_channels.begin(); channel != ycbcr_channels.end(); ++channel)
		{
			SingalChannelImageDoG(*channel, vector_sigmas, dog_channels);
		}
		
		// LBP on all DOG image channels
		//std::cout<<"dog channels nums: "<<dog_channels.size()<<std::endl;
		cv::Mat sample_hist_vector;  
		for(std::vector<cv::Mat>::iterator channel = dog_channels.begin(); channel != dog_channels.end(); ++channel)
		{
			//std::cout<<"rows: "<<sample_hist_vector.rows<<"cols: "<<sample_hist_vector.cols<<std::endl;
			cv::Mat channel_lbp_hist_cell;
			lbp_cell.computeLBPFeatureVector(*channel, channel_lbp_hist_cell, LBP::Mode::RIU2);
			sample_hist_vector = mergeCols(sample_hist_vector, channel_lbp_hist_cell);
			cv::Mat channel_lbp_hist_full;
			lbp_full.computeLBPFeatureVector(*channel, channel_lbp_hist_full, LBP::Mode::RIU2);
			sample_hist_vector = mergeCols(sample_hist_vector, channel_lbp_hist_full);
		}
		// push back sample
		data.push_back(sample_hist_vector);

		/* prepare label feature vector */
		if((*it).find("fake") != string::npos)  
		{
			label.push_back(0);
		}
		else if((*it).find("living") != string::npos)  
		{
			label.push_back(1);
		}

		++progressBar;
		progressBar.display();
	}
	progressBar.done();

}

void parseArguments(const int& argc, const char* const* argv,
					int& resize, int& cellsize, std::string& train_list, std::string& test_list, Action& action)
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
			else if( strcmp( argv[i], "-a" ) == 0 ) {
				if( strcmp( argv[i+1], "test" ) == 0 )
				{
					action = TEST;
				}
				else if( strcmp( argv[i+1], "train" ) == 0 )
				{
					action = TRAIN;
				}
				else
				{
					std::cout<<"[ERROR]: Invalid action. Action can only be 'train' or 'test'."<<std::endl;
					std::cout<<"\t\t\t Execution aborted."<<std::endl;
					exit( 1 );
				}
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

void printHelp() {
	std::cout << "\nUsage: ./lbp [options]" << std::endl;
	std::cout << "\nOptions:" << std::endl;
	std::cout << "\t-r <int> - Target resize size (default=96)" << std::endl;
	std::cout << "\t-c <int> - Desired cell size for LBP extraction (default=16)" << std::endl;
	std::cout << "\t-t <string> - Path to a txt file contains a list of training data" << std::endl;
	std::cout << "\t-v <string> - Path to a txt file contains a list of testing data" << std::endl;
	std::cout << "\t-a <string> - action parameters: train/test. ('train' will run both train and test, 'test' will only run test)" << std::endl;
}
