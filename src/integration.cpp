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
#include <fstream>

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

    std::cout<<"merged train data samples: "<<train_data.rows<<" len: "<<train_data.cols<<std::endl;
    std::cout<<"merged train label samples: "<<train_label.rows<<" len: "<<train_label.cols<<std::endl; 
    std::cout<<"[Note]: Writting merged train data HDF5..."<<std::endl;
    saveMatToHDF5(train_data.clone(), train_label.clone(), std::string("merged_train"));


    
    std::cout<<"[Note]: Preparing testing data..."<<std::endl;
    prepareData(dog_lbp_data_dir, ofm_data_dir, gray_img_dir,
    			net1, net2, net3,
    			lbp_test_filelist, test,
    			test_data, test_label);

    std::cout<<"merged test data samples: "<<test_data.rows<<" len: "<<test_data.cols<<std::endl;
    std::cout<<"merged test label samples: "<<test_label.rows<<" len: "<<test_label.cols<<std::endl; 
    std::cout<<"[Note]: Writting merged test data HDF5..."<<std::endl;
    saveMatToHDF5(test_data.clone(), test_label.clone(), std::string("merged_test"));
    

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

	std::ofstream file;
	file.open("sample+softmax.txt", std::ios_base::app); 
	// setup progress bar
    int total_ticks = filelist.size();
	ProgressBar progressBar(total_ticks, 70, '=', '-');
    for(std::vector<std::string>::const_iterator it = filelist.begin(); it != filelist.end(); ++it)
    {
    	// proceed only if same sample exisits in all three data directory
    	if(exists( dir2+subdir+(*it) ) && exists( dir3+subdir+(*it) ) )
    	{
            //////////////////////////////////////////////////////////// 3 way inference merge ////////////////////////////////////////////////////////
    		//file << *it <<": \n";
    		/*
    		// model 1 sample preparation 
    		//std::cout<<"[Note]: Preparing model 1 data..."<<std::endl;
    		cv::Mat model1_image, model1_image_resized, model1_input_feature;
    		model1_image = cv::imread( dir1+subdir+(*it), cv::IMREAD_COLOR);
    		cv::resize(model1_image, model1_image_resized, cv::Size(64, 64));
    		if( dog_lbp_extraction(model1_image_resized, model1_input_feature, 64, 8) )
    		{
    			continue;
    		}
    		

    		// model 2 sample preparation
    		//std::cout<<"[Note]: Preparing model 2 data..."<<std::endl;
    		cv::Mat model2_image, model2_image_resized, model2_input_feature;
    		model2_image = cv::imread( dir2+subdir+(*it), cv::IMREAD_GRAYSCALE);
    		cv::resize(model2_image, model2_image_resized, cv::Size(64, 64));
    		if( ofm_extraction(model2_image_resized, model2_input_feature, 64, 8) )
    		{
    			continue;
    		}

    		// model 3 sample preparation
    		//std::cout<<"[Note]: Preparing model 3 data..."<<std::endl;
    		cv::Mat model3_image, model3_image_resized, model3_input_feature;
    		model3_image = cv::imread( dir3+subdir+(*it), cv::IMREAD_GRAYSCALE);
    		cv::resize(model3_image, model3_image_resized, cv::Size(64, 64));
    		if( gray_lbp_extraction(model3_image_resized, model3_input_feature, 64, 8) )
    		{
    			continue;
    		}
            */

    		/* merging all inference result into one feature vector */
    		//cv::Mat sample_feature_vector;
    		//sample_feature_vector = mergeCols(model2_input_feature, model3_input_feature);

    		/*
    		// model 1 inference
    		//std::cout<<"[Note]: Model1 input vector shape: rows: "<<model1_input_feature.rows<<" cols: "<<model1_input_feature.cols<<std::endl;
			net1.setInput(model1_input_feature);
			cv::Mat model1_result = net1.forward(std::string("prob"));
			//cv::Mat model1_result = net1.getParam(net1.getLayerId(std::string("ip3")));
			sample_feature_vector = mergeCols(sample_feature_vector, model1_result);
			

    		// model 2 inference
    		//std::cout<<"[Note]: Model2 input vector shape: rows: "<<model2_input_feature.rows<<" cols: "<<model2_input_feature.cols<<std::endl;
			net2.setInput(model2_input_feature);
			cv::Mat model2_result = net2.forward(std::string("relu1"));
			//cv::Mat model2_result = net2.getParam(net2.getLayerId(std::string("ip3")));
			sample_feature_vector = mergeCols(sample_feature_vector, model2_result);
			file << "model2 softmax: " << model2_result<<" ";

    		// model 3 inference
    		//std::cout<<"[Note]: Model3 input vector shape: rows: "<<model3_input_feature.rows<<" cols: "<<model3_input_feature.cols<<std::endl;
			net3.setInput(model3_input_feature);
			cv::Mat model3_result = net3.forward(std::string("relu1"));
			//cv::Mat model3_result = net3.getParam(net3.getLayerId(std::string("ip3")));
			sample_feature_vector = mergeCols(sample_feature_vector, model3_result);
			file << "model3 softmax: " << model3_result<<"\n ";
			*/

            //////////////////////////////////////////////////////////// 3 way inference merge ////////////////////////////////////////////////////////

            //////////////////////////////////////////////////////////// auto encoder /////////////////////////////////////////////////////////////////
            // model 1 sample preparation 
            //std::cout<<"[Note]: Preparing model 1 data..."<<std::endl;
            cv::Mat model1_image, model1_image_resized, model1_input_feature;
            model1_image = cv::imread( dir1+subdir+(*it), cv::IMREAD_COLOR);
            cv::resize(model1_image, model1_image_resized, cv::Size(64, 64));
            cv::Mat response;
            double sigma1 = 0.5, sigma2 = 1;
            findFrequencyReponse(model1_image_resized, response, sigma1, sigma2);  // response ready
            

            // model 2 sample preparation
            //std::cout<<"[Note]: Preparing model 2 data..."<<std::endl;
            cv::Mat model2_image, model2_image_resized, model2_input_feature;
            int cellsize=8;
            model2_image = cv::imread( dir2+subdir+(*it), cv::IMREAD_GRAYSCALE);
            cv::resize(model2_image, model2_image_resized, cv::Size(64, 64));
            std::vector<double> hist;
            cv::Mat raw_lbp_feature;
            getFaceLBPHist(model2_image_resized, hist, cellsize);
            raw_lbp_feature.push_back(hist);
            cv::transpose(raw_lbp_feature, raw_lbp_feature);                      // raw_lbp_feature ready


            cv::Mat sample_feature_vector;
            // autoencoder
            // model 1 autoencode
            //std::cout<<"[Note]: Model1 input vector shape: rows: "<<model1_input_feature.rows<<" cols: "<<model1_input_feature.cols<<std::endl;
            net1.setInput(response);
            cv::Mat model1_result = net1.forward(std::string("bottleneck"));
            sample_feature_vector = mergeCols(sample_feature_vector, model1_result);
            

            // model 2 autoencode
            //std::cout<<"[Note]: Model2 input vector shape: rows: "<<model2_input_feature.rows<<" cols: "<<model2_input_feature.cols<<std::endl;
            net2.setInput(raw_lbp_feature);
            cv::Mat model2_result = net2.forward(std::string("bottleneck"));
            sample_feature_vector = mergeCols(sample_feature_vector, model2_result);


            //////////////////////////////////////////////////////////// auto encoder /////////////////////////////////////////////////////////////////

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

   if(1) 
    {
    	if (action==train) {

			std::cout<<"normalize train data feature to N(0,1)..."<<std::endl;
            vector<float> feature_mean, feature_stddev; 


            cv::Mat col0 = data.col(0).clone(); 

            for(int c=0; c<data.cols; c++ ) 
            {
               cv::Mat col1 = data.col(c).clone(); 
               cv::Scalar mean,stddev; //0:1st channel, 1:2nd channel and 2:3rd channel
               cv::meanStdDev(col1, mean, stddev); 
               //cout << "col = " << c << " mean = " << mean << " stddev = " << stddev << endl; 
               col1 = (1.0/stddev[0])*(col1 - mean[0]); 
               //data.col(c)  = col1;
               col1.copyTo(data.col(c));

               feature_mean.push_back(mean[0]); 
               feature_stddev.push_back(stddev[0]); 
            } 

            /*FILE *fp0 = fopen("debug.txt", "wt");
            cv::Mat col2 = data.col(0).clone(); 
            for (int r=0; r<data.rows;r++) 
               fprintf(fp0,"%f %f\n", col0.at<float>(r,0),col2.at<float>(r,0)); 
            fclose(fp0);*/  


			// save to xml
			FILE *fp = fopen("mean_stddev.txt", "wt");
            for(int c=0; c<data.cols; c++ )
			{
				fprintf(fp,"%f %f\n",feature_mean[c],feature_stddev[c]); 
			}
            fclose(fp); 
        }
        else
        {
			std::cout<<"normalize test data feature to N(0,1)..."<<std::endl;
            vector<float> feature_mean, feature_stddev; 

			FILE *fp = fopen("mean_stddev.txt", "rt");
           
            for(int c=0; c<data.cols; c++ ) 
            {
               float mean, stddev; 
               fscanf(fp,"%f %f\n",&mean,&stddev); 
               //cout << "col = " << c << " mean = " << mean << " stddev = " << stddev << endl; 
               feature_mean.push_back(mean); 
               feature_stddev.push_back(stddev); 
            }
            fclose(fp); 

            for(int c=0; c<data.cols; c++ ) 
            {
               cv::Mat col1 = data.col(c).clone(); 
               col1 = (1.0/feature_stddev[c])*(col1 - feature_mean[c]); 
               //data.col(c)  = col1;
               col1.copyTo(data.col(c));

            } 

        }
    }

    file.close();
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


int ofm_extraction(cv::Mat& img, cv::Mat& feature_vector, const int resize, const int cell_size)
{
	if(img.empty())
	{
		std::cout<<"[ERROR]: Input image for OFM_LBP extraction is empty. Skipping this sample..."<< std::endl;
		return 1; 
	}

	LBP lbp_cell(cv::Size(cell_size, cell_size));
	LBP lbp_full(cv::Size(resize, resize));

	lbp_cell.computeLBPFeatureVector(img, feature_vector, LBP::Mode::RIU1);
	cv::Mat full_lbp_hist;
	lbp_full.computeLBPFeatureVector(img, full_lbp_hist, LBP::Mode::RIU1);
	feature_vector = mergeCols(feature_vector, full_lbp_hist);

	// normalize data with pre-calculated means and stdvs
	std::ifstream infile("thefile.txt");
	float mean, stdv;
	float *dataOfFeatureVector=(float *)feature_vector.data;
	for(int col = 0; col < feature_vector.cols; ++col)
	{
		infile >> mean >> stdv;
		dataOfFeatureVector[col] = (dataOfFeatureVector[col]-mean) / stdv;
	}

	return 0;
}


int gray_lbp_extraction(cv::Mat& img, cv::Mat& feature_vector, const int resize, const int cell_size)
{
	if(img.empty())
	{
		std::cout<<"[ERROR]: Input image for OFM_LBP extraction is empty. Skipping this sample..."<< std::endl;
		return 1; 
	}

	LBP lbp_cell(cv::Size(cell_size, cell_size));
	LBP lbp_full(cv::Size(resize, resize));

	lbp_cell.computeLBPFeatureVector(img, feature_vector, LBP::Mode::RIU1);
	cv::Mat full_lbp_hist;
	lbp_full.computeLBPFeatureVector(img, full_lbp_hist, LBP::Mode::RIU1);
	feature_vector = mergeCols(feature_vector, full_lbp_hist);

	// this section normalized the feature vector 
	// normalize data with pre-calculated means and stdvs
	std::ifstream infile("thefile.txt");
	float mean, stdv;
	// offset the file, starting from 3770
	int offset;
	for(offset = 0 ; offset < 3770; ++offset)
	{
		infile >> mean >> stdv;
	}

	float *dataOfFeatureVector=(float *)feature_vector.data;
	for(int col = offset; col < feature_vector.cols; ++col)
	{
		infile >> mean >> stdv;
		dataOfFeatureVector[col] = (dataOfFeatureVector[col]-mean) / stdv;
	}

	return 0;
}

void findFrequencyReponse(cv::Mat& resizedImg, cv::Mat& response, const double sigma1, const double sigma2)
{
    std::vector<float> fftThresh = {0,10,20,50,100,200,500}; 
    //std::vector<float> fftThresh = {10}; 
   
    int normGrayMeanEnable = 0; 
    float normGrayMean = 180; 
   
    if(normGrayMeanEnable==1) 
    {
      // normalize
      cv::Scalar mean, std;
      cv::meanStdDev(resizedImg, mean, std);
      std::cout <<"origin image mean: " << mean[0] << std::endl;
      resizedImg.convertTo(resizedImg, CV_32F);
      resizedImg = resizedImg / cv::Mat(resizedImg.rows, resizedImg.cols, CV_32F, cv::Scalar(mean[0]/180));
    }
    else if (normGrayMeanEnable==2) 
    {
      cv::normalize(resizedImg,  resizedImg, 0, 255, cv::NORM_MINMAX);
      //std::cout <<"normed: " << resizedImg << std::endl;
    }
    cv::Mat sample_dog_fft;
    sample_dog_fft = FFTDOG(resizedImg,0,sigma1, sigma2);

    //std::cout<<fr_bin<<std::endl;
    float DCEnergy = cv::sum(cv::sum(resizedImg))[0]; 
    float totalEnergy = cv::sum(cv::sum(sample_dog_fft))[0]; 

    // checking dft result image
    //std::cout<<"dftimage size:"<< sample_dog_fft.rows<< "," << sample_dog_fft.cols <<std::endl;
    //cv::imshow( "dft image", sample_dog_fft );                   // Show our image inside it.
    //cv::waitKey(0); 
    
    // note: please use square image: eg. 64*64 
    CV_Assert(sample_dog_fft.rows == sample_dog_fft.cols);
    // Allocate frequencey historgram bin: contains sample_dog_fft.rows number of bins
    // representing sample_dog_fft.row number of discrete frequency
    int max_radius = sample_dog_fft.rows/2;
    int min_radius = 0;

   for(int t=0; t<fftThresh.size(); t++) 
   {

    cv::Mat fr_bin = cv::Mat::zeros(cv::Size(max_radius + 1 - min_radius, 1), CV_32F); // add one extra bin to store all high frequency 
    cv::Mat bin_cnt = cv::Mat::zeros(cv::Size(max_radius + 1 - min_radius, 1), CV_32F);
    // define the center of the image
    std::pair<float, float> center(0.5*(sample_dog_fft.rows-1), 0.5*(sample_dog_fft.cols-1)); 
    //std::cout<<fr_bin.rows<< " , "<< fr_bin.cols<<std::endl;

    for(int r = 0; r < sample_dog_fft.rows; ++r)
    {
        for(int c = 0; c < sample_dog_fft.cols; ++c)
        {   
            int distance = sqrt( std::pow((r-center.first), 2) + std::pow((c-center.second), 2) );
            //std::cout<< "ditance:" << distance << " pairs: " << std::pow((r-center.first), 2) <<", " <<std::pow((c-center.second), 2) << std::endl;
            if(distance >= max_radius) // high frequency
            {
                //bin_cnt.at<float>(max_radius-min_radius) += 1;
                if(sample_dog_fft.at<float>(r, c)>fftThresh[t])
                    fr_bin.at<float>(max_radius-min_radius) += sample_dog_fft.at<float>(r, c);
            } 
            else if (distance >= min_radius)  // low frequency 
            {
                //bin_cnt.at<float>(distance-min_radius) += 1;
                if(sample_dog_fft.at<float>(r, c)>fftThresh[t])
                    fr_bin.at<float>(distance-min_radius) += sample_dog_fft.at<float>(r, c);
            }
            // else is to low to be considered
        }
    }

    
    //fr_bin = fr_bin / cv::Mat(fr_bin.rows, fr_bin.cols, CV_32FC1, cv::Scalar(cv::sum( fr_bin )[0]));
    //fr_bin = fr_bin / bin_cnt; 
    /*for(int b=0; b<max_radius + 1 - min_radius;b++) 
    {
        if(bin_cnt.at<float>(b)>0)
            fr_bin.at<float>(b) =  fr_bin.at<float>(b) /bin_cnt.at<float>(b); 
    }*/
    //cv::normalize(fr_bin,fr_bin, 0, 1, cv::NORM_L2); // Transform the matrix with float values into a
   
    //fr_bin = fr_bin / cv::Mat(fr_bin.rows, fr_bin.cols, CV_32FC1, totalEnergy); 
    fr_bin = fr_bin/(totalEnergy-DCEnergy); 
    cv::Mat fr_bin2  = fr_bin.colRange(1, fr_bin.cols);  
    response = mergeCols(response, fr_bin2);   
    //std::cout << "t = " << t << " thresh = " << fftThresh[t] <<  " fr_bin size " << fr_bin.size() << " fr_bin2 size " << fr_bin2.size() << " response size = " << response.size() << std::endl;   

   } // for t 

    //std::cout<<fr_bin<<std::endl;
    //std::cout<< cv::Mat(fr_bin.rows, fr_bin.cols, CV_32FC1, cv::Scalar(cv::sum( bin_cnt )[0])) <<std::endl;
    //std::cout<<fr_bin.rows<< " , "<< fr_bin.cols<<std::endl;    
} 


// FFT on DOG Features
cv::Mat FFTDOG(cv::Mat srcImg, int DOGEnable, const double sigma1, const double sigma2)
{
    cv::Mat XF1, XF2, DXF, output;
    int size1, size2;

    if(DOGEnable) 
        {
          // Filter Sizes
          if(DOGEnable==1) 
          {
        size1 = 2 * (int)(3*sigma1) + 3;
        size2 = 2 * (int)(3*sigma2) + 3;
          }
          else 
          { size1 = 9; 
            size2 = 9; 
          }   
          // Gaussian Filter
      cv::GaussianBlur(srcImg, XF1, cv::Size(size1, size1), sigma1, sigma1, cv::BORDER_REPLICATE);
      cv::GaussianBlur(srcImg, XF2, cv::Size(size2, size2), sigma2, sigma2, cv::BORDER_REPLICATE);
      // Difference
      DXF = XF1 - XF2;
        }
    else 
          DXF = srcImg;

        DXF.convertTo(DXF, CV_32F);

        int hanningWindowEnable = 1; 
        if(hanningWindowEnable)
        {
          //apply Hanning window before FFT 
          cv::Mat hann1t = cv::Mat(cv::Size(DXF.rows,1), CV_32F, cv::Scalar(0));
          cv::Mat hann2t = cv::Mat(cv::Size(1,DXF.cols), CV_32F, cv::Scalar(0)); 
          for (int i = 0; i < hann1t.cols; i++)
            hann1t.at<float > (0, i) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann1t.cols - 1)));
          for (int i = 0; i < hann2t.rows; i++)
             hann2t.at<float > (i, 0) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann2t.rows - 1)));
          cv::Mat hann2d = hann2t * hann1t; 
          DXF = DXF.mul(hann2d);
        }

    // Discrete Fourier Transform
    // pad the image for best performance
    cv::Mat PadDXF;
    int row = cv::getOptimalDFTSize( DXF.rows );
    int col = cv::getOptimalDFTSize( DXF.cols );
    cv::copyMakeBorder(DXF, PadDXF, 0, row - DXF.rows, 0, col - DXF.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    // create 2 channel Mat to store real and imaginary part of the dft result
    cv::Mat planes[] = {cv::Mat_<float>(PadDXF), cv::Mat::zeros(PadDXF.size(), CV_32F)};
    cv::Mat complex;
    cv::merge(planes, 2, complex);
    cv::dft(complex, complex);
    // calcualte the dft respond magnitude
    split(complex, planes);
    cv::magnitude(planes[0], planes[1], planes[0]);
    cv::Mat dftMag = planes[0];
    
    //dftMag += cv::Scalar::all(1);                    // switch to logarithmic scale
    //log(dftMag, dftMag);

    // crop the spectrum, if it has an odd number of rows or columns
    dftMag = dftMag(cv::Rect(0, 0, dftMag.cols & -2, dftMag.rows & -2));
    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = dftMag.cols/2;
    int cy = dftMag.rows/2;
    cv::Mat q0(dftMag, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    cv::Mat q1(dftMag, cv::Rect(cx, 0, cx, cy));  // Top-Right
    cv::Mat q2(dftMag, cv::Rect(0, cy, cx, cy));  // Bottom-Left
    cv::Mat q3(dftMag, cv::Rect(cx, cy, cx, cy)); // Bottom-Right
    cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
    
    //cv::normalize(dftMag, dftMag, 0, 1, cv::NORM_MINMAX); // Transform the matrix with float values into a

    return dftMag;
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