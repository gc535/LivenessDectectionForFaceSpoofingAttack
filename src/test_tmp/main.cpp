
#include <iostream>
#include <unistd.h>
#include "ProgressBar.hpp"

int main()
{
	int total = 1000;
    ProgressBar progressBar(total, 70, '=', '-');

    for (int i = 0; i < total; i++) {
        ++progressBar; // record the tick

        usleep(2000); // simulate work
        progressBar.display();
        /*
        // display the bar only at certain steps
        if (i % 10 == 0)
            
    	*/
    }

    // tell the bar to finish
    progressBar.done();

    std::cout << "Done!" << std::endl;

    cv::Mat empty;
    cv::Mat img(1,1,CV_8U,cvScalar(0));
    std::cout<< empty.cols<<"_"<<img.cols<< std::endl;
    cv::Mat merged;
    cv::hconcat(empty, img, merged);
    std::cout<< merged.cols<< std::endl;
    
    return 0;
}