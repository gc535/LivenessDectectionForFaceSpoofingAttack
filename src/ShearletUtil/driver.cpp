#include <helper.hpp>
#include <iostream>
#include <shear.hpp>


int main()
{   

    /*
    std::vector<int> m(4, 32);
    std::vector<int> num(4, 2);
    int L = 256;

    std::vector<std::vector<cv::Mat> > w_s;
    for(int j = 0; j < num.size(); ++j)
    {
        int n1 = m[j];
        int level = num[j];

        cv::Mat x11, y11, x12, y12, F1;
        Shearlet::gen_x_y_cordinates(n1, x11, y11, x12, y12, F1);
        int N = 2*n1;
        int M = pow(2, level) + 2;

        cv::Mat wf = Shearlet::windowing(cv::Mat::ones(N, 1, CV_32S), pow(2, level), 1);
        cv::Mat w = cv::Mat::zeros(n1, n1, CV_32F);   
        std::vector<cv::Mat> directional_w_s;
        for(int k = 0; k < M; ++k)
        {
            cv::Mat indexed_col = wf.col(k);
            cv::Mat temp = indexed_col * cv::Mat::ones(1, N/2, CV_32F);
            cv::Mat result = Shearlet::rec_from_pol(temp,n1,x11,y11,x12,y12,F1);
            directional_w_s.push_back(result);
            w = w + result;
        }
        for(int k = 0; k < M; ++k)
        {
            directional_w_s[k] = Mat_sqrt( directional_w_s[k].mul(1/w) );
            cv::Mat input_real = directional_w_s[k]
            cv::Mat input_img = cv::Mat::zeros(input_real.row, input_real.cols, CV_64F);
            input_real.convertTo(input_real, CV_64F);
            
            cv::Mat ifftshift_real, ifftshift_img;
            ifft2shift(input_real, input_img, ifftshift_real, ifftshift_img);
            cv::Mat ifft2_real, ifft2_img;
            ifft2(ifftshift_real, ifftshift_img, ifft2_real, ifft2_img, false);
            cv::Mat fft2_real, fft2_img;
            fft2shift(ifft2_real, ifft2_img, fft2_real, fft2_img);

            directional_w_s[k] = fft2_real;
        }

        w_s.push_back(directional_w_s);
    }
    */

    
    
    //std::cout<< "x11: " << x11 << std::endl;
    //std::cout<< "y11: " << y11 << std::endl;
    //std::cout<< "x12: " << x12 << std::endl;
    //std::cout<< "y12: " << y12 << std::endl;
    //std::cout<< "F1: " << F1 << std::endl;
    
    
    
    // std::cout << "wf:" << wf << std::endl;
    
    //std::cout << "indexed_col rows:" << indexed_col.rows << ", cols: " << indexed_col.cols << std::endl;
    
    //std::cout << "temp rows:" << temp.rows << ", cols: " << temp.cols << std::endl;
    //std::cout << "temp:" << temp << std::endl;
    
    //std::cout << "result rows:" << result.rows << ", cols: " << result.cols << std::endl;
    //std::cout << "result:" << result << std::endl;
    //Shearlet::shearing_filters_Myer(m, num, 256);

    /////////////////////////////////////////
    std::vector<int> m(4, 32);
    std::vector<int> num(4, 2);
    int L = 256;
    std::vector<std::vector<cv::Mat> > w_s;
    for(int j = 0; j < 4; ++j)
    {
        std::vector<cv::Mat> j_row;
        for(int k = 0; k < 6; ++k)
        {
            cv::Mat k_mat;
            std::string filename = std::to_string(j+1)+"_"+std::to_string(k+1)+".txt";
            readFile2Mat(k_mat, filename);
            j_row.push_back(k_mat);
        }
        w_s.push_back(j_row);
    }

    //std::cout << w_s[0][0] << std::endl;

    std::vector<std::vector<std::pair<cv::Mat, cv::Mat> > > result_shears;
    std::vector<std::vector<std::pair<cv::Mat, cv::Mat> > > dshears;
    for(int j = 0; j < num.size(); ++j)
    {
        int r = w_s[j][0].rows;
        int c = w_s[j][0].cols;
        int n = w_s[j].size();
        cv::Mat w_real = cv::Mat::zeros(L, L, CV_64F);
        cv::Mat w_img = cv::Mat::zeros(L, L,  CV_64F);

        std::vector<std::pair<cv::Mat, cv::Mat> > directional_shears;
        for(int k = 0; k < n; ++k)
        {
            cv::Mat shear_real = cv::Mat::zeros(L, L, CV_64F);
            cv::Mat shear_img;
            // set value in ROI
            cv::Mat ROI = shear_real(cv::Rect(0, 0, r, c));
            for(int x = 0; x < r; ++x)
            {
                for(int y = 0; y < c; ++y)
                {
                    ROI.at<double>(x, y) = w_s[j][k].at<double>(x, y);
                }
            }
            //std::cout << shear << std::endl;
            cv::Mat tmp_real, tmp_img;
            fft2(shear_real, shear_img, tmp_real, tmp_img, false);
            std::pair<cv::Mat, cv::Mat> shear_pair(tmp_real, tmp_img);
            directional_shears.push_back(shear_pair);

            cv::Mat tmp_square_real, tmp_square_img;
            Shearlet::complexMul(tmp_real, tmp_img, tmp_real, tmp_img,
                                 tmp_square_real, tmp_square_img);
            Shearlet::complexAdd(w_real, w_img, tmp_square_real, tmp_square_img,
                                 w_real, w_img);
        }
        result_shears.push_back(directional_shears);
        
        cv::Mat z_real = cv::Mat::zeros(L, L, CV_64F);
        cv::Mat z_img = cv::Mat::zeros(L, L, CV_64F);

        std::vector<std::pair<cv::Mat, cv::Mat> > directional_dshears;
        cv::Mat place_holder;
        // pushing back place holder for k=0
        directional_dshears.push_back(std::pair<cv::Mat, cv::Mat>(place_holder, place_holder) );
        for(int k = 1; k < n-1; ++k)
        {
            // dshear{j}(:,:,k) = sqrt(1./w.*shear{j}(:,:,k).^2);
            // 1./w.
            cv::Mat tmp1_real, tmp1_img;
            cv::Mat one_real = cv::Mat::ones(w_real.rows, w_real.cols, CV_64F);
            cv::Mat one_img = cv::Mat::zeros(w_img.rows, w_img.cols, CV_64F);
            Shearlet::complexDiv(one_real, one_img, w_real, w_img, 
                                 tmp1_real, tmp1_img);

            // shear{j}(:,:,k).^2
            cv::Mat tmp2_real, tmp2_img;
            Shearlet::complexMul(directional_shears[k].first, directional_shears[k].second, 
                                 directional_shears[k].first, directional_shears[k].second,
                                 tmp2_real, tmp2_img);

            // 1./w.  *   shear{j}(:,:,k).^2
            cv::Mat tmp3_real, tmp3_img;
            Shearlet::complexMul(tmp1_real, tmp1_img, tmp2_real, tmp2_img,
                                 tmp3_real, tmp3_img);

            // sqrt( tmp result)
            cv::Mat tmp4_real, tmp4_img;
            Shearlet::complexSqrt(tmp3_real, tmp3_img, tmp4_real, tmp4_img);
            
            std::pair<cv::Mat, cv::Mat> dshear_pair(tmp4_real, tmp4_real);
            directional_dshears.push_back(dshear_pair);

            // z = z+dshear{j}(:,:,k).^2
            //       dshear{j}(:,:,k).^2 is tmp3 (tmp4 = sqrt(tmp3))
            Shearlet::complexAdd(z_real, z_img, tmp3_real, tmp3_img,
                                 z_real, z_img);
            writeMatToFile(z_real, std::string("z_real.txt"));
            writeMatToFile(z_img, std::string("z_img.txt"));
            exit(0);
        }
        // pushing back place holder for k=n-1
        directional_dshears.push_back(std::pair<cv::Mat, cv::Mat>(place_holder, place_holder) ); 
        //std::cout << "size of d"
        
        /* 
          processing the first and last directional_dshear pairs:
            dshear{j}(:,:,1) = sqrt([zeros(L/2) ones(L/2); ones(L/2) zeros(L/2)].*s);
            dshear{j}(:,:,n) = sqrt([ones(L/2) zeros(L/2); zeros(L/2) ones(L/2)].*s);
        */
        cv::Mat s_real = cv::Mat::ones(z_real.rows, z_real.cols, CV_64F) - z_real;
        cv::Mat s_img = cv::Mat::zeros(z_img.rows, z_img.cols, CV_64F) - z_img;
        directional_dshears[0] = std::pair<cv::Mat, cv::Mat>(s_real, s_img);
        directional_dshears[n-1] = std::pair<cv::Mat, cv::Mat>(s_real, s_img);
        for(int r = 0; r < s_real.rows; ++r)
        {
            for(int c = 0; c < s_real.cols; ++c)
            {   
                // first quadrant
                if(c < L/2 && r < L/2)
                {
                    directional_dshears[0].first.at<double>(r, c) = 0;
                    directional_dshears[0].second.at<double>(r, c) = 0;
                }
                // second quadrant
                else if(c >= L/2 and r < L/2)
                {
                    directional_dshears[1].first.at<double>(r, c) = 0;
                    directional_dshears[1].second.at<double>(r, c) = 0;
                }
                // third quadrant
                else if(c < L/2 and r >= L/2)
                {
                    directional_dshears[1].first.at<double>(r, c) = 0;
                    directional_dshears[1].second.at<double>(r, c) = 0;
                }
                // fourth quadrant
                else
                {
                    directional_dshears[0].first.at<double>(r, c) = 0;
                    directional_dshears[0].second.at<double>(r, c) = 0;
                }
            }
        }
        Shearlet::complexSqrt(directional_dshears[0].first, directional_dshears[0].second,
                              directional_dshears[0].first, directional_dshears[0].second);
        Shearlet::complexSqrt(directional_dshears[1].first, directional_dshears[1].second,
                              directional_dshears[1].first, directional_dshears[1].second);
        
        // push back entire directional_dshears to dshears
        dshears.push_back(directional_dshears);
    }
    
    return 0;
}
