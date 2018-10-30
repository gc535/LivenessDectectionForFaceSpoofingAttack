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
	std::string dog_lbp_data_path = "", ofm_data_path = "", gray_img_path = "";
	int resize = 96, cell_size = 16;
	ParseArgument(argc, argv, 
				  model_1, checkpoint_1, 
				  model_2, checkpoint_2, 
				  model_3, checkpoint_3, 
				  resize, cell_size, 
				  dog_lbp_data_path, ofm_data_path, gray_img_path);


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

    if(getFilelist( dog_lbp_data_path+std::string("/train/"), lbp_train_filelist))
    {
    	std::cout<<"[ERROR]: Cannot get a filelist from: "<<dog_lbp_data_path+std::string("/train/")<<std::endl;
    }
    if(getFilelist( dog_lbp_data_path+std::string("/test/"), lbp_test_filelist))
    {
    	std::cout<<"[ERROR]: Cannot get a filelist from: "<<dog_lbp_data_path+std::string("/test/")<<std::endl;
    }


    /* prepararing integrated data by merging inference result from all three model calculation */
    std::cout<<"[Note]: Starting merging data..."<<std::endl;
    for(std::vector<std::string>::iterator it = lbp_train_filelist.begin(); it != lbp_train_filelist.end(); ++it)
    {
    	// proceed only if same sample exisits in all three data directory
    	if(exists( ofm_data_path+std::string("/train/")+(*it) ) && exists( gray_img_path+std::string("/train/")+(*it) ) )
    	{
    		// model 1 sample preparation 
    		std::cout<<"[Note]: Preparing model 1 data..."<<std::endl;
    		cv::Mat model1_image, model1_image_resized, model1_input_feature;
    		model1_image = cv::imread(dog_lbp_data_path+std::string("/train/")+(*it), cv::IMREAD_COLOR);
    		cv::resize(model1_image, model1_image_resized, cv::Size(resize, resize));
    		if( dog_lbp_extraction(model1_image_resized, model1_input_feature, resize, cell_size) )
    		{
    			continue;
    		}

    		// model 2 sample preparation
    		std::cout<<"[Note]: Preparing model 2 data..."<<std::endl;
    		cv::Mat model2_image, model2_image_resized, model2_input_feature;
    		model2_image = cv::imread(ofm_data_path+std::string("/train/")+(*it), cv::IMREAD_GRAYSCALE);
    		cv::resize(model2_image, model2_image_resized, cv::Size(resize, resize));
    		if( ofm_extraction(model2_image_resized, model2_input_feature, cell_size) )
    		{
    			continue;
    		}

    		// model 3 sample preparation
    		std::cout<<"[Note]: Preparing model 3 data..."<<std::endl;
    		cv::Mat model3_image, model3_image_resized, model3_input_feature;
    		model3_image = cv::imread(gray_img_path+std::string("/train/")+(*it), cv::IMREAD_GRAYSCALE);
    		cv::resize(model3_image, model3_image_resized, cv::Size(resize, resize));
    		if( gray_lbp_extraction(model3_image_resized, model3_input_feature, cell_size) )
    		{
    			continue;
    		}

    		// model 1 inference
    		std::cout<<"[Note]: Starting model 1 inference..."<<std::endl;
			 net1.setInput(model1_input_feature);
			 net1.forward();
			 cv::Mat model1_result = net1.getParam(net1.getLayerId(std::string("ip3")));
			 cv::Mat model1_prob = net1.getParam(net1.getLayerId(std::string("prob")));


    		// model 2 inference



    		// model 3 inference
    	}
    }

    /*
    if (parser.get<bool>("opencl"))
    {
        net.setPreferableTarget(DNN_TARGET_OPENCL);
    }
    Mat img = imread(imageFile);
    if (img.empty())
    {
        std::cerr << "Can't read image from the file: " << imageFile << std::endl;
        exit(-1);
    }
    //GoogLeNet accepts only 224x224 BGR-images
    Mat inputBlob = blobFromImage(img, 1.0f, Size(224, 224),
                                  Scalar(104, 117, 123), false);   //Convert Mat to batch of images
    net.setInput(inputBlob, "data");        //set the network input
    Mat prob = net.forward("prob");         //compute output
    cv::TickMeter t;
    for (int i = 0; i < 10; i++)
    {
        CV_TRACE_REGION("forward");
        net.setInput(inputBlob, "data");        //set the network input
        t.start();
        prob = net.forward("prob");                          //compute output
        t.stop();
    }
    int classId;
    double classProb;
    getMaxClass(prob, &classId, &classProb);//find the best class
    std::vector<String> classNames = readClassNames();
    std::cout << "Best class: #" << classId << " '" << classNames.at(classId) << "'" << std::endl;
    std::cout << "Probability: " << classProb * 100 << "%" << std::endl;
    std::cout << "Time: " << (double)t.getTimeMilli() / t.getCounter() << " ms (average from " << t.getCounter() << " iterations)" << std::endl;
    */

    return 0;
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
				   std::string& dog_lbp_data_path, std::string& ofm_data_path, std::string& gray_img_path)
{
	if( argc < 11 ) {
		printHelp();
		exit( 1 );
	}
	else if( argc >= 11 ) {
		// process arguments
		for( int i = 1; i < argc - 1; i++ ) {
			if( strcmp( argv[i], "-d1" ) == 0 ) {
				dog_lbp_data_path = argv[i + 1];
				i++;
			}
			else if( strcmp( argv[i], "-d2" ) == 0 ) {
				ofm_data_path = argv[i + 1];
				i++;
			}
			else if( strcmp( argv[i], "-d3" ) == 0 ) {
				gray_img_path = argv[i + 1];
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