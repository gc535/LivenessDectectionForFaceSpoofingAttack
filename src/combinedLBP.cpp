#include <combinedLBP.hpp>


int main(int argc, char** argv)
{
    int resize = 64;
    int cellsize = 8;
    std::string train_list = "", test_list = "";
    parseArguments(argc, argv,
                   resize, cellsize, train_list, test_list);

    // prepare train data
    Data data(train_list, Data::Action::TRAIN);
    cv::Mat train_data, train_label;
    data.DataPreparation(combineFeatures, train_data, train_label, "combined_lbp", resize, cellsize);

    // prepare test data
    data.update(test_list, Data::Action::TEST);
    cv::Mat test_data, test_label;
    data.DataPreparation(combineFeatures, test_data, test_label, "combined_lbp", resize, cellsize);

    return 0;
}

// calculate the everage enegery response at each discrete frequency
void combineFeatures(cv::Mat& data, cv::Mat& label, const std::vector<std::string>& filelist, 
                const int resize, const int cellsize)
{
    cv::Mat srcImg;
    cv::Mat resizedImg;
    int total_ticks = filelist.size();
    ProgressBar progressBar(total_ticks, 70, '=', '-');
    for(std::vector<std::string>::const_iterator it = filelist.begin(); it != filelist.end(); ++it)
    {
        //std::cout << "filename = " << *it << std::endl; 
        //srcImg = cv::imread("/home/ubuntu/Desktop/data/NUAA/test/fake-230", cv::IMREAD_COLOR);  //debug
        srcImg = cv::imread(*it, cv::IMREAD_COLOR);
        cv::resize(srcImg, resizedImg, cv::Size(resize, resize));

        // set sigmas
        const std::vector<cv::Vec2d> vector_sigmas = { cv::Vec2d(0.5, 1), cv::Vec2d(1, 2), cv::Vec2d(0.5,2)};

        // dog_lbp
        cv::Mat sample_hist_doglbp;
        DOG_LBP(resizedImg, sample_hist_doglbp, vector_sigmas, resize, cellsize);

        // raw_lbp
        cv::Mat sample_hist_rawlbp;
        std::vector<double> hist;
        getFaceLBPHist(resizedImg, hist, cellsize);
        sample_hist_rawlbp.push_back(hist);
        cv::transpose(sample_hist_rawlbp, sample_hist_rawlbp); 
        cv::Mat sample_hist = mergeCols(sample_hist_doglbp, sample_hist_rawlbp);
        // push back sample
        data.push_back(sample_hist);

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

void DOG_LBP(cv::Mat& srcImg, cv::Mat& sample_hist_vector, const std::vector<cv::Vec2d> vector_sigmas, const int resize, const int cellsize)
{

    LBP lbp_cell(cv::Size(cellsize, cellsize));    // setup the lbp extractor
    LBP lbp_full(cv::Size(resize, resize));        // setup the lbp extractor

    // hsv
    cv::Mat hsv_image;
    cv::cvtColor(srcImg, hsv_image, cv::COLOR_RGB2HSV);
    const std::vector<cv::Mat> hsv_channels = splitChannels(hsv_image);
    std::vector<cv::Mat> dog_channels;
    for(std::vector<cv::Mat>::const_iterator channel = hsv_channels.begin(); channel != hsv_channels.end(); ++channel)
    {
        SingalChannelImageDoG(*channel, vector_sigmas, dog_channels);
    }
    
    // ycbcr
    cv::Mat ycbcr_image;
    cv::cvtColor(srcImg, ycbcr_image, cv::COLOR_RGB2YCrCb);
    const std::vector<cv::Mat> ycbcr_channels = splitChannels(ycbcr_image);
    for(std::vector<cv::Mat>::const_iterator channel = ycbcr_channels.begin(); channel != ycbcr_channels.end(); ++channel)
    {
        SingalChannelImageDoG(*channel, vector_sigmas, dog_channels);
    }
    
    // LBP on all DOG image channels
    //std::cout<<"dog channels nums: "<<dog_channels.size()<<std::endl;  
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

    // Normalize
    float sum = 0;
    for (cv::MatIterator_<float> it = sample_hist_vector.begin<float>(); it != sample_hist_vector.end<float>(); ++it) {
        sum += *it;
    }

    for (cv::MatIterator_<float> it = sample_hist_vector.begin<float>(); it != sample_hist_vector.end<float>(); ++it) {
        *it = (*it) / sum;
    }
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
    std::cout << "\t-r  <int> - Target resize size (default=64)" << std::endl;
    std::cout << "\t-c  <int> - Desired cell size for LBP extraction (default=8)" << std::endl;
    std::cout << "\t-t <string> - Path to a txt file contains a list of training data" << std::endl;
    std::cout << "\t-v <string> - Path to a txt file contains a list of testing data" << std::endl;
}

/*
/home/ubuntu/Desktop/data/NUAA/test/fake-230
/home/ubuntu/Desktop/data/NUAA/test/living-340
*/
