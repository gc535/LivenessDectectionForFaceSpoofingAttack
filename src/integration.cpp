#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/trace.hpp>

#include <integration.hpp>
#include <SingleChannelDOG.hpp>
#include <LBP.hpp>
#include <ProgressBar.hpp>
#include <Util.hpp>


#include <stdlib.h>
#include <string>
#include <vector>
#include <iostream>


int main(int argc, char** argv)
{

	std::string model_1 = "", checkpoint_1 = "";
	std::string model_2 = "", checkpoint_2 = "";
	std::string model_3 = "", checkpoint_3 = "";
	std::string dog_lbp_data_dir = "", ofm_data_dir = "", gray_img_dir = "";
	int resize = 96, cell_size = 16;
	ParseArgument(argc, argv, 
				  model_1, checkpoint_1, 
				  model_2, checkpoint_2, 
				  model_3, checkpoint_3, 
				  resize, cell_size, 
				  dog_lbp_data_dir, ofm_data_dir, gray_img_dir);


	/* loading network models */
	cv::dnn::Net net1, net2, net3;
    try 
    {
    	std::cerr << "[NOTE]: Loading model and binary for net1: " << std::endl;
        net1 = cv::dnn::readNetFromCaffe(model_1, checkpoint_1);
    }
    catch (cv::Exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        if (net1.empty())
        {
            std::cerr << "[ERROR]: Can't load network by using the following files: " << std::endl;
            std::cerr << "\t\tprototxt:   " << model_1 << std::endl;
            std::cerr << "\t\tcaffemodel: " << checkpoint_1 << std::endl;
            std::cerr << "\t\tplease make sure Pre-trained model exists in the specified directory" << std::endl;

            exit(-1);
        }
    }
    net1.setPreferableTarget(CPU);
    try 
    {
    	std::cerr << "[NOTE]: Loading model and binary for net2: " << std::endl;
        net2 = cv::dnn::readNetFromCaffe(model_2, checkpoint_2);
    }
    catch (cv::Exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        if (net2.empty())
        {
            std::cerr << "[ERROR]: Can't load network by using the following files: " << std::endl;
            std::cerr << "\t\tprototxt:   " << model_2 << std::endl;
            std::cerr << "\t\tcaffemodel: " << checkpoint_2 << std::endl;
            std::cerr << "\t\tplease make sure Pre-trained model exists in the specified directory" << std::endl;

            exit(-1);
        }
    }
    net2.setPreferableTarget(CPU);
    try 
    {
    	std::cerr << "[NOTE]: Loading model and binary for net3: " << std::endl;
        net3 = cv::dnn::readNetFromCaffe(model_3, checkpoint_3);
    }
    catch (cv::Exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        if (net3.empty())
        {
            std::cerr << "[ERROR]: Can't load network by using the following files: " << std::endl;
            std::cerr << "\t\tprototxt:   " << model_3 << std::endl;
            std::cerr << "\t\tcaffemodel: " << checkpoint_3 << std::endl;
            std::cerr << "\t\tplease make sure Pre-trained model exists in the specified directory" << std::endl;

            exit(-1);
        }
    }
    net3.setPreferableTarget(CPU);

    /* prepare traning/testing file list */
    std::vector<std::string> lbp_train_filelist, lbp_test_filelist;
    std::vector<std::string> ofm_train_filelist, ofm_test_filelist;

    if(getFilelist( dog_lbp_data_dir+std::string("/train/"), lbp_train_filelist))
    {
    	std::cout<<"[ERROR]: Cannot get a filelist from: "<<dog_lbp_data_dir+std::string("/train/")<<std::endl;
    }
    if(getFilelist( dog_lbp_data_dir+std::string("/test/"), lbp_test_filelist))
    {
    	std::cout<<"[ERROR]: Cannot get a filelist from: "<<dog_lbp_data_dir+std::string("/test/")<<std::endl;
    }


    /* prepararing integrated data by merging inference result from all three models' calculation */
    std::cout<<"[Note]: Starting merging data..."<<std::endl;
    // setup data and label matrix for collecting sample data
    cv::Mat train_data, train_label, test_data, test_label;
    std::cout<<"[Note]: Preparing training data..."<<std::endl;
    prepareData(dog_lbp_data_dir, ofm_data_dir, gray_img_dir,
    			net1, net2, net3,
    			lbp_train_filelist, train,
    			train_data, train_label);

    std::cout<<"[Note]: Preparing testing data..."<<std::endl;
    prepareData(dog_lbp_data_dir, ofm_data_dir, gray_img_dir,
    			net1, net2, net3,
    			lbp_test_filelist, test,
    			test_data, test_label);
    
    return 0;
}


void prepareData(const std::string& dir1, const std::string& dir2, const std::string& dir3, 
				 cv::dnn::Net& net1, cv::dnn::Net& net2, cv::dnn::Net& net3,
				 const std::vector<std::string>& filelist, Action action, 
				 cv::Mat& data, cv::Mat& label)
{
	// determine subdirectory from action
	std::string subdir = action==train ? "/train/" : "/test/";

	if(filelist.empty())
	{
		std::cout<<"[ERROR]: "<<subdir <<" filelist empty. please check."<< std::endl;
		exit(1);
	}
	// setup progress bar
    int total_ticks = filelist.size();
	ProgressBar progressBar(total_ticks, 70, '=', '-');
    for(std::vector<std::string>::const_iterator it = filelist.begin(); it != filelist.end(); ++it)
    {
    	// proceed only if same sample exisits in all three data directory
    	if(exists( dir2+subdir+(*it) ) && exists( dir3+subdir+(*it) ) )
    	{
    		// model 1 sample preparation 
    		std::cout<<"[Note]: Preparing model 1 data..."<<std::endl;
    		cv::Mat model1_image, model1_image_resized, model1_input_feature;
    		model1_image = cv::imread( dir1+subdir+(*it), cv::IMREAD_COLOR);
    		cv::resize(model1_image, model1_image_resized, cv::Size(64, 64));
    		if( dog_lbp_extraction(model1_image_resized, model1_input_feature, 64, 8) )
    		{
    			continue;
    		}

    		// model 2 sample preparation
    		std::cout<<"[Note]: Preparing model 2 data..."<<std::endl;
    		cv::Mat model2_image, model2_image_resized, model2_input_feature;
    		model2_image = cv::imread( dir2+subdir+(*it), cv::IMREAD_GRAYSCALE);
    		cv::resize(model2_image, model2_image_resized, cv::Size(64, 64));
    		if( ofm_extraction(model2_image_resized, model2_input_feature, 8) )
    		{
    			continue;
    		}

    		// model 3 sample preparation
    		std::cout<<"[Note]: Preparing model 3 data..."<<std::endl;
    		cv::Mat model3_image, model3_image_resized, model3_input_feature;
    		model3_image = cv::imread( dir3+subdir+(*it), cv::IMREAD_GRAYSCALE);
    		cv::resize(model3_image, model3_image_resized, cv::Size(64, 64));
    		if( gray_lbp_extraction(model3_image_resized, model3_input_feature, 8) )
    		{
    			continue;
    		}

    		/* merging all inference result into one feature vector */
    		cv::Mat sample_feature_vector;
    		// model 1 inference
    		std::cout<<"[Note]: Model1 input vector shape: rows: "<<model1_input_feature.rows<<" cols: "<<model1_input_feature.cols<<std::endl;
			net1.setInput(model1_input_feature);
			cv::Mat model1_prob = net1.forward();
			//std::cout<<"[Note]: Ip3 layers id: "<<net1.getLayerId(std::string("ip3"))<<std::endl;
			cv::Mat model1_result = net1.getParam(net1.getLayerId(std::string("ip3")));
			sample_feature_vector = mergeCols(sample_feature_vector, model1_result);

    		// model 2 inference
    		std::cout<<"[Note]: Model2 input vector shape: rows: "<<model2_input_feature.rows<<" cols: "<<model2_input_feature.cols<<std::endl;
			net2.setInput(model2_input_feature);
			cv::Mat model2_prob = net2.forward();
			//std::cout<<"[Note]: Ip3 layers id: "<<net1.getLayerId(std::string("ip3"))<<std::endl;
			cv::Mat model2_result = net2.getParam(net2.getLayerId(std::string("ip3")));
			sample_feature_vector = mergeCols(sample_feature_vector, model2_result);

    		// model 3 inference
    		std::cout<<"[Note]: Model3 input vector shape: rows: "<<model3_input_feature.rows<<" cols: "<<model3_input_feature.cols<<std::endl;
			net3.setInput(model3_input_feature);
			cv::Mat model3_prob = net3.forward();
			//std::cout<<"[Note]: Ip3 layers id: "<<net1.getLayerId(std::string("ip3"))<<std::endl;
			cv::Mat model3_result = net3.getParam(net3.getLayerId(std::string("ip3")));
			sample_feature_vector = mergeCols(sample_feature_vector, model3_result);

			// push back sample
			data.push_back(sample_feature_vector);

			// push back label
			if((*it).find("fake") != string::npos)  
			{
				label.push_back(0);
			}
			else if((*it).find("living") != string::npos)  
			{
				label.push_back(1);
			}
		}
		++progressBar;
		progressBar.display();
    }
    progressBar.done();
}


int dog_lbp_extraction(cv::Mat& img, cv::Mat& feature_vector, const int resize, const int cell_size)
{
	if(img.empty())
	{
		std::cout<<"[ERROR]: Input image for DOG_LBP extraction is empty. Skipping this sample..."<< std::endl;
		return 1; 
	}
	/* prepare feature vector for every sample*/
	std::vector<cv::Mat> dog_channels;
	const std::vector<cv::Vec2d> vector_sigmas = { cv::Vec2d(0.5, 1), cv::Vec2d(1, 2), cv::Vec2d(0.5,2)};
	
	LBP lbp_cell(cv::Size(cell_size, cell_size));    // setup the lbp extractor
	LBP lbp_full(cv::Size(resize, resize));		   // setup the lbp extractor

	// hsv
	cv::Mat hsv_image;
	cv::cvtColor(img, hsv_image, cv::COLOR_RGB2HSV);
	const std::vector<cv::Mat> hsv_channels = splitChannels(hsv_image);
	for(std::vector<cv::Mat>::const_iterator channel = hsv_channels.begin(); channel != hsv_channels.end(); ++channel)
	{
		SingalChannelImageDoG(*channel, vector_sigmas, dog_channels);
	}
	
	
	// ycbcr
	cv::Mat ycbcr_image;
	cv::cvtColor(img, ycbcr_image, cv::COLOR_RGB2YCrCb);
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
		feature_vector = mergeCols(feature_vector, channel_lbp_hist_cell);
		cv::Mat channel_lbp_hist_full;
		lbp_full.computeLBPFeatureVector(*channel, channel_lbp_hist_full, LBP::Mode::RIU2);
		feature_vector = mergeCols(feature_vector, channel_lbp_hist_full);
	}

	return 0;  // return 0 if success
}


int ofm_extraction(cv::Mat& img, cv::Mat& feature_vector, const int cell_size)
{
	if(img.empty())
	{
		std::cout<<"[ERROR]: Input image for OFM_LBP extraction is empty. Skipping this sample..."<< std::endl;
		return 1; 
	}

	LBP lbp(cv::Size(cell_size, cell_size));
	lbp.computeLBPFeatureVector(img, feature_vector, LBP::Mode::RIU2);

	return 0;
}


int gray_lbp_extraction(cv::Mat& img, cv::Mat& feature_vector, const int cell_size)
{
	if(img.empty())
	{
		std::cout<<"[ERROR]: Input image for OFM_LBP extraction is empty. Skipping this sample..."<< std::endl;
		return 1; 
	}

	LBP lbp(cv::Size(cell_size, cell_size));
	lbp.computeLBPFeatureVector(img, feature_vector, LBP::Mode::RIU2);

	return 0;
}


void ParseArgument(const int& argc, const char* const* argv, 
				   std::string& model_1, std::string& checkpoint_1, 
				   std::string& model_2, std::string& checkpoint_2, 
				   std::string& model_3, std::string& checkpoint_3,
				   int& resize, int& cell_size,
				   std::string& dog_lbp_data_dir, std::string& ofm_data_dir, std::string& gray_img_dir)
{
	if( argc < 11 ) {
		printHelp();
		exit( 1 );
	}
	else if( argc >= 11 ) {
		// process arguments
		for( int i = 1; i < argc - 1; i++ ) {
			if( strcmp( argv[i], "-d1" ) == 0 ) {
				dog_lbp_data_dir = argv[i + 1];
				i++;
			}
			else if( strcmp( argv[i], "-d2" ) == 0 ) {
				ofm_data_dir = argv[i + 1];
				i++;
			}
			else if( strcmp( argv[i], "-d3" ) == 0 ) {
				gray_img_dir = argv[i + 1];
				i++;
			}
			else if( strcmp( argv[i], "-r" ) == 0 ) {
				resize = atoi( argv[i + 1] );
				i++;
			}
			else if( strcmp( argv[i], "-c" ) == 0 ) {
				cell_size = atoi( argv[i + 1] );
				i++;
			}
			else if( strcmp( argv[i], "-m1" ) == 0 ) {
				model_1 = argv[i + 1];
				i++;
			}
			else if( strcmp( argv[i], "-b1" ) == 0 ) {
				checkpoint_1 = argv[i + 1];
				i++;
			}
			else if( strcmp( argv[i], "-m2" ) == 0 ) {
				model_2 = argv[i + 1];
				i++;
			}
			else if( strcmp( argv[i], "-b2" ) == 0 ) {
				checkpoint_2 = argv[i + 1];
				i++;
			}
			else if( strcmp( argv[i], "-m3" ) == 0 ) {
				model_3 = argv[i + 1];
				i++;
			}
			else if( strcmp( argv[i], "-b3" ) == 0 ) {
				checkpoint_3 = argv[i + 1];
				i++;
			}
			else {
				std::cerr << "invalid argument: \'" << argv[i] << "\'\n";
				printHelp();
				exit( 1 );
			}
		}
	}
}

void printHelp() {
	std::cout << "\nUsage: ./combine [options]" << std::endl;
	std::cout << "\nOptions:" << std::endl;
	std::cout << "\t-m1 <string> - LBP model path (*.prototxt)" << std::endl;
	std::cout << "\t-b1 <string> - Binary LBP model parameters (*.caffemodel)" << std::endl;
	std::cout << "\t-m2 <string> - OFM model path (*.prototxt)" << std::endl;
	std::cout << "\t-b2 <string> - Binary OFM model parameters (*.caffemodel)" << std::endl;
	std::cout << "\t-m3 <string> - Gray image LBP model path (*.prototxt)" << std::endl;
	std::cout << "\t-b3 <string> - Binary gray image LBP model parameters (*.caffemodel)" << std::endl;
	std::cout << "\t-r  <int> - Target resize size (default=96)" << std::endl;
	std::cout << "\t-c  <int> - Desired cell size for LBP extraction (default=16)" << std::endl;
	std::cout << "\t-d1 <string> - path to image folder for model 1." << std::endl;
	std::cout << "\t-d2 <string> - path to image folder for model 2." << std::endl;
	std::cout << "\t-d3 <string> - path to image folder for model 3." << std::endl;
	std::cout << "\t \t \tData root should contain a two sub folder: 'train' and 'test" << std::endl;
}