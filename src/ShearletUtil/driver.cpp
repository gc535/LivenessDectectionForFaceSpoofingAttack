#include <helper.hpp>
#include <iostream>

int main()
{   
    
    cv::Mat x11, y11, x12, y12, F1;
    int n1 = 32;
    Shearlet::gen_x_y_cordinates(n1, x11, y11, x12, y12, F1);
    
    //std::cout<< "x11: " << x11 << std::endl;
    //std::cout<< "y11: " << y11 << std::endl;
    //std::cout<< "x12: " << x12 << std::endl;
    //std::cout<< "y12: " << y12 << std::endl;
    //std::cout<< "F1: " << F1 << std::endl;
    
    int N = 2*n1;
    int level = 2;
    int M = pow(2, level) + 2;
    cv::Mat wf = Shearlet::windowing(cv::Mat::ones(N, 1, CV_32S), pow(2, level), 1);
    // std::cout << "wf:" << wf << std::endl;
    


    //std::vector<int> m(4, 32);
    //std::vector<int> num(4, 2);
    cv::Mat w = cv::Mat::zeros(n1, n1, CV_32F);

    int k = 0;  // k: 1-6 in MATLAB
    cv::Mat indexed_col = wf.col(k);
    //std::cout << "indexed_col rows:" << indexed_col.rows << ", cols: " << indexed_col.cols << std::endl;
    cv::Mat temp = indexed_col * cv::Mat::ones(1, N/2, CV_32F);
    //std::cout << "temp rows:" << temp.rows << ", cols: " << temp.cols << std::endl;
    //std::cout << "temp:" << temp << std::endl;
    cv::Mat result = Shearlet::rec_from_pol(temp,n1,x11,y11,x12,y12,F1);
    //std::cout << "result rows:" << result.rows << ", cols: " << result.cols << std::endl;
    //std::cout << "result:" << result << std::endl;
    //Shearlet::shearing_filters_Myer(m, num, 256);
    
    return 0;
}
