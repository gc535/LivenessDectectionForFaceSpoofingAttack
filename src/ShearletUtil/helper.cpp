#include <helper.hpp>

namespace Shearlet
{

void gen_x_y_cordinates(int n, cv::Mat& x1n, cv::Mat& y1n, cv::Mat& x2n, cv::Mat& y2n, cv::Mat& D)
{
    /* 
      This function is a c++ implementation of:
      https://github.com/lymhust/Fig2_Fig3_Fig4-for-NR-IQA-with-Shearlet-Transform-and-DNN/blob/master/Shearlet-Transform/gen_x_y_cordinates.m
    */
    n += 1;
    cv::Mat x1 = cv::Mat::zeros(n, n, CV_32S);
    cv::Mat y1 = cv::Mat::zeros(n, n, CV_32S);
    cv::Mat x2 = cv::Mat::zeros(n, n, CV_32S);
    cv::Mat y2 = cv::Mat::zeros(n, n, CV_32S);
    cv::Mat m1 = cv::Mat::zeros(1, n, CV_32F);

    for(int i = 0; i < n; ++i)
    {
        int y0 = 1, x0 = i+1, x_n = n-i, y_n = n;
        int flag = x_n == x0 ? 1 : 0;
        if(x_n != x0)
	    {
	        m1.at<float>(i) = 1.0*(y_n - y0)/(x_n - x0);
	    }
        std::vector<double> line_space = linespace(x0, x_n, n);
        cv::Mat xt(line_space);
        for(int j = 0; j < n; ++j)
        {
            if(!flag)
            {
	            y1.at<int>(i, j) = round(m1.at<float>(i) * (xt.at<double>(j) - x0) + y0);  // wrapped by round operation
                x1.at<int>(i, j) = round((xt.at<double>(j)));        // wrapped by round operation
                x2.at<int>(i, j) = y1.at<int>(i, j);
                y2.at<int>(i, j) = x1.at<int>(i, j);
            }
            else
            {
                x1.at<int>(i, j) = (n - 1)/2 + 1;
                y1.at<int>(i, j) = j+1;
                x2.at<int>(i, j) = j+1;
                y2.at<int>(i, j) = (n - 1)/2 + 1;
            }
        }
    }

    n -= 1;
    x1n = cv::Mat::zeros(n, n, CV_32S);
    y1n = cv::Mat::zeros(n, n, CV_32S);
    x2n = cv::Mat::zeros(n, n, CV_32S);
    y2n = cv::Mat::zeros(n, n, CV_32S);
    
    for(int i = 0; i < n; ++i)
    {
        for(int j = 0; j < n; ++j)
        {
            x1n.at<int>(i, j) = x1.at<int>(i, j);
            y1n.at<int>(i, j) = y1.at<int>(i, j);
            x2n.at<int>(i, j) = x2.at<int>(i+1, j);
            y2n.at<int>(i, j) = y2.at<int>(i+1, j);
        }
    }

    y2n.at<int>(n-1, 0) = n;
    cv::flip(x1n, x1n, 0);
    cv::flip(y2n, y2n, 0);
    //std::cout<< "y1n: " << y1n << std::endl;
    D = avg_pol(n, x1n, y1n, x2n, y2n);
}


std::vector<double> linespace(double start, double end, int num_sample)
{
    std::vector<double> vector;
    double delta = (end - start) / (num_sample - 1); 
    for(int cnt = 0; cnt < num_sample; ++cnt)
    {
        vector.push_back(start + cnt*delta);
    }
    return vector;
}

cv::Mat avg_pol(int L, const cv::Mat& x1, const cv::Mat& y1, const cv::Mat& x2, const cv::Mat& y2)
{   
    // This function generates the matrix that contains the number
    // of times the polar grid points go through the rectangular grid 
    // point i,j
    //
    // Input: L is the order of the block matrix
    //
    // Output: D is the common grid point values
    cv::Mat D = cv::Mat::zeros(L, L, CV_32S);
    for(int i = 0; i < L; ++i)
    {
        for(int j = 0; j < L; ++j)
        {
            D.at<int>(y1.at<int>(i, j)-1, x1.at<int>(i, j)-1) = D.at<int>(y1.at<int>(i, j)-1, x1.at<int>(i, j)-1) + 1;
        }
    }

    for(int i = 0; i < L; ++i)
    {
        for(int j = 0; j < L; ++j)
        {
            D.at<int>(y2.at<int>(i, j)-1, x2.at<int>(i, j)-1) = D.at<int>(y2.at<int>(i, j)-1, x2.at<int>(i, j)-1) + 1;
        }
    }
    return D;
}

cv::Mat rec_from_pol(cv::Mat& l, int n, cv::Mat& x1, cv::Mat& y1, cv::Mat& x2, cv::Mat& y2, cv::Mat& D)
{
    // https://github.com/lymhust/Fig2_Fig3_Fig4-for-NR-IQA-with-Shearlet-Transform-and-DNN/blob/master/Shearlet-Transform/rec_from_pol.m
    // This funcion re-assembles the radial slice into block.
    //
    // Inputs: l is the radial slice matrix
    //         n is the order of the matrix that is to be re-assembled
    //         x1,y1,x2,y2 are the polar coordinates generated from function
    //                gen_x_y_cord
    //         D is the matrix containing common polar grid points
    // Output: C is the re-assembled block matrix 
    cv::Mat C = cv::Mat::zeros(n, n, CV_32F);
    int option = 0;
    if(option == 1)
    {
        for(int i = 0; i < n; ++i)
        {
            for(int j = 0; j < n; ++j)
            {
                C.at<float>(y1.at<int>(i, j)-1, x1.at<int>(i, j)-1) = l.at<float>(i, j);
                C.at<float>(y2.at<int>(i, j)-1, x2.at<int>(i, j)-1) = l.at<float>(i+n, j);
            }
        }
        
    }
    else
    {
        for(int i = 0; i < n; ++i)
        {
            for(int j = 0; j < n; ++j)
            {
                C.at<float>(y1.at<int>(i, j)-1, x1.at<int>(i, j)-1) = C.at<float>(y1.at<int>(i, j)-1, x1.at<int>(i, j)-1) + l.at<float>(i, j);
                C.at<float>(y2.at<int>(i, j)-1, x2.at<int>(i, j)-1) = C.at<float>(y2.at<int>(i, j)-1, x2.at<int>(i, j)-1) + l.at<float>(i+n, j);
            }
        }
    }
    // average common radial grid values
    cv::Mat output;
    cv::divide(C, D, output, 1, C.type()); // scale = 1
    
    //std::cout << "output rows:" << output.rows << ", cols: " << output.cols << std::endl;
    //std::cout << "output:" << output << std::endl;
    return output;
}

cv::Mat windowing(cv::Mat x, const int L, const int c)
{   
    // https://github.com/lymhust/Fig2_Fig3_Fig4-for-NR-IQA-with-Shearlet-Transform-and-DNN/blob/master/Shearlet-Transform/windowing.m
    // This function computes the Meyer window for
    // L decompositions.
    // Inputs:  x - signal to window
    //          L - number of bandpass filters
    //          c - scaling parameter (e.g: c = 1)
    // Outputs: y - y(:,k) is windowed signal x with bandpass k
    int N = x.rows;
    cv::Mat y = cv::Mat::zeros(N, (L+2), CV_32F);  // output
    int T = ceil(N/(float)(L+2));
    
    cv::Mat g = cv::Mat::zeros(1, 2*T, CV_32F);
    for(int i = 0; i < g.cols; ++i)
    {
        float n = (-1) * T/2.0 + (i+1) - 1;   // since i starts from 1 in .m file
        float test = meyer_wind(c*n/T);
        g.at<float>(i) = test;
    }

    for(int i = 1; i < y.cols; ++i)
    {
        int index = 0;
        float k = -1.0*T/2;
        while(k <= 1.5*T-1)
        {
	        int in_sig = floor( fmod( k+(i+1-1)*T, N) ) + 1;  // since j starts from 2 in .m file
            y.at<float>(in_sig-1, i) = g.at<float>(index) * x.at<int>(in_sig-1);
	        k = k+1;
            index++;
        }
    }
    //std::cout << y << std::endl;

    cv::Mat row_sum;
    cv::reduce(y, row_sum, 1, cv::REDUCE_SUM, CV_32F);
    for(int i = 0; i < y.rows; ++i)
    {
        y.at<float>(i, 0) = std::abs(1 - row_sum.at<float>(i));
    }

    return y;
}

float meyer_wind(const float q)
{
    // https://github.com/lymhust/Fig2_Fig3_Fig4-for-NR-IQA-with-Shearlet-Transform-and-DNN/blob/master/Shearlet-Transform/meyer_wind.m
    // This function computes the Meyer window function
    if(-1.0/3+1.0/2 < q && q<1.0/3+1.0/2)
    {
       return 1;
    }
    else if( (1.0/3+1.0/2<=q & q<=2.0/3+1.0/2) | (-2.0/3+1.0/2<=q & q<=1.0/3+1.0/2) )
    {
        float w = 3 * std::abs(q-1.0/2) - 1;
        float z = pow(w, 4) * (35 - 84*w + 70*pow(w, 2) - 20*pow(w, 3));
        return pow( cos(M_PI/2.0*(z)), 2);
    }
    else
    {
        return 0;
    }
}


std::vector<std::vector<std::pair<cv::Mat, cv::Mat> > > shearing_filters_Myer(std::vector<int>& m, std::vector<int>& num, const int L)
{
    // This function computes the shearing filters (wedge shaped) using the Meyer window
    // function.
    //
    // Inputs: m - size of shearing filter matrix desired, m = [m(1),...,m(N)] where
    //             each entry m(j) determines size of shearing filter matrix at scale j. 
    //         num - the parameter determining the number of directions. 
    //               num = [num(1),...,num(N)] where each entry num(j)
    //               determines the number of directions at scale j.  
    //               num(j) ---> 2^(num(j)) + 2 directions.
    //         L - size of the input image ; L by L input image. 
    //
    //
    // Outputs: dshear{j}(:,:,k) - m(j) by m(j) shearing  filter matrix at orientation
    //                             k and scale j.  
    //
    //
    // For example, dshear=shearing_filters_Myer([100 100 180 180],[3 3 4 4],L);
    // produces cell array 'dshear' consisting of 
    //          10 shearing filters (100 by 100) at scale j = 1 (coarse scale)  
    //          10 shearing filters (100 by 100) at scale j = 2 
    //          18 shearing filters (180 by 180) at scale j = 3 
    //          18 shearing filters (180 by 180) at scale j = 4 (fine scale) 

    /* constructing w_s */
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
            directional_w_s[k] = Shearlet::Mat_sqrt( directional_w_s[k].mul(1/w) );
            cv::Mat input_real = directional_w_s[k];
            cv::Mat input_img = cv::Mat::zeros(input_real.rows, input_real.cols, CV_64F);
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
    /* END constructing w_s */

    /* constructing dshears */
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
        }
        // pushing back place holder for k=n-1
        directional_dshears.push_back(std::pair<cv::Mat, cv::Mat>(place_holder, place_holder) ); 
        
         
        //  processing the first and last directional_dshear pairs:
        //    MATLAB: dshear{j}(:,:,1) = sqrt([zeros(L/2) ones(L/2); ones(L/2) zeros(L/2)].*s);
        //    MATLAB: dshear{j}(:,:,n) = sqrt([ones(L/2) zeros(L/2); zeros(L/2) ones(L/2)].*s);
        cv::Mat s_real = cv::Mat::ones(z_real.rows, z_real.cols, CV_64F) - z_real;
        cv::Mat s_img = cv::Mat::zeros(z_img.rows, z_img.cols, CV_64F) - z_img;
        directional_dshears[0] = std::pair<cv::Mat, cv::Mat>(s_real.clone(), s_img.clone());
        directional_dshears[n-1] = std::pair<cv::Mat, cv::Mat>(s_real.clone(), s_img.clone());
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
                else if(c >= L/2 && r < L/2)
                {
                    directional_dshears[n-1].first.at<double>(r, c) = 0;
                    directional_dshears[n-1].second.at<double>(r, c) = 0;
                }
                // third quadrant
                else if(c < L/2 && r >= L/2)
                {
                    directional_dshears[n-1].first.at<double>(r, c) = 0;
                    directional_dshears[n-1].second.at<double>(r, c) = 0;
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
        Shearlet::complexSqrt(directional_dshears[n-1].first, directional_dshears[n-1].second,
                              directional_dshears[n-1].first, directional_dshears[n-1].second);
        
        // push back entire directional_dshears to dshears
        dshears.push_back(directional_dshears);
    }
    /* END constructing dshears */

    return dshears;
}

cv::Mat Mat_sqrt(cv::Mat input)
{
    input.convertTo(input, CV_64F);
    cv::Mat result = cv::Mat::zeros(input.rows, input.cols, CV_64F);
    for(int r = 0; r < input.rows; ++r)
    {
        for(int c = 0; c < input.cols; ++c)
        {
            result.at<double>(r, c) = sqrt( input.at<double>(r, c) );
        }
    }
    return result;
}


void complexDiv(cv::Mat& src1_real, cv::Mat& src1_img, cv::Mat& src2_real, cv::Mat& src2_img,
                cv::Mat& out_real, cv::Mat& out_img)
{   
    /* 
     this function calculate the division between two complex number:
        a + bj     (a + bj)(c - dj)     ac - adj + bcj + bd
       -------- = ------------------ = -------------------- 
        c + dj     (c + dj)(c - dj)         c^2 + d^2

        ac + bd        bc - ad
    = ----------- + j-----------
       c^2 + d^2      c^2 + d^2

    */
    cv::Mat den = src2_real.mul(src2_real) + src2_img.mul(src2_img); 
    cv::Mat real_num = src1_real.mul(src2_real) + src1_img.mul(src2_img);
    cv::Mat img_num = src1_img.mul(src2_real) - src1_real.mul(src2_img);

    cv::divide(real_num, den, out_real, 1, CV_64F);
    cv::divide(img_num, den, out_img, 1, CV_64F);
}

void complexMul(cv::Mat& src1_real, cv::Mat& src1_img, cv::Mat& src2_real, cv::Mat& src2_img,
                cv::Mat& out_real, cv::Mat& out_img)
{   
    /* 
     this function calculate the division between two complex number:
       (a + bj)(c + dj)  =  ac + adj + bcj - bd
    
     = (ac - bd) + j(bc + ad)

    */
    out_real = src1_real.mul(src2_real) - src1_img.mul(src2_img);
    out_img = src1_img.mul(src2_real) + src1_real.mul(src2_img);
}


void complexAdd(cv::Mat& src1_real, cv::Mat& src1_img, cv::Mat& src2_real, cv::Mat&src2_img,
                cv::Mat& out_real, cv::Mat& out_img)
{
    out_real = src1_real + src2_real;
    out_img = src1_img + src2_img;
}

void complexSqrt(cv::Mat& real, cv::Mat& img, cv::Mat& result_real, cv::Mat& result_img)
{
    /*
        assume (p + qi)^2 = a + bi, then we have
            1. p^2 - q^2 = a
            2. 2pq = b
        
        Solving this two equation we will have:
             sqrt(a + sqrt(a^2 + b^2))
        p = ---------------------------
                    sqrt(2)

        then we can calculate:
             b
        q = ----
             2p   
    */
    cv::Mat q_num = Mat_sqrt(real + Mat_sqrt(real.mul(real) + img.mul(img)));
    result_real = q_num / sqrt(2);
    cv::divide(img, result_real, result_img, 0.5, CV_64F);
}

}
