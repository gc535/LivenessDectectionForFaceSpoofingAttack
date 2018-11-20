#include <lbp_hist.hpp>
#include <RawLBP.hpp>

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
    data.DataPreparation(RawLBP, train_data, train_label, "raw_lbp", resize, cellsize);

    // prepare test data
    data.update(test_list, Data::Action::TEST);
    cv::Mat test_data, test_label;
    data.DataPreparation(RawLBP, test_data, test_label, "raw_lbp", resize, cellsize);



}


void RawLBP(cv::Mat& data, cv::Mat& label, const std::vector<std::string>& filelist, 
            const int resize, const int cellsize)
{
    cv::Mat srcImg;
    cv::Mat resizedImg;
    std::vector<double> hist;
    

    int total_ticks = filelist.size();
    ProgressBar progressBar(total_ticks, 70, '=', '-');
    for(std::vector<std::string>::const_iterator it = filelist.begin(); it != filelist.end(); ++it)
    {
        srcImg = cv::imread(*it, cv::IMREAD_COLOR);
        cv::resize(srcImg, resizedImg, cv::Size(resize, resize));
    
        // hsv
        cv::Mat hsv_image;
        cv::cvtColor(resizedImg, hsv_image, cv::COLOR_RGB2HSV);
        std::vector<cv::Mat> hsv_channels = splitChannels(hsv_image);
        cv::Mat sample_hist;
        for(std::vector<cv::Mat>::iterator channel = hsv_channels.begin(); channel != hsv_channels.end(); ++channel)
        {
            cv::Mat channel_hist;
            getFaceLBPHist(*channel, hist, cellsize);
            channel_hist.push_back(hist);
            sample_hist = mergeCols(sample_hist, channel_hist);
        }
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

/*
/home/ubuntu/Desktop/d/combined/test/living-540.jpg
/home/ubuntu/Desktop/d/combined/test/fake-1153.jpg
/home/ubuntu/Desktop/d/combined/test/fake-356.jpg
/home/ubuntu/Desktop/d/combined/test/living-1102.jpg
/home/ubuntu/Desktop/d/combined/test/living-798.jpg
/home/ubuntu/Desktop/d/combined/test/living-901.jpg
/home/ubuntu/Desktop/d/combined/test/living-1239.jpg



*/