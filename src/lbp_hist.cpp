#include "lbp_hist.hpp"
#include "ELBP.hpp"

using namespace lbp;
static void spatial_Histogram(LBP lbp, Mat lbpImg, int width, int height, vector<double> &histC, int block = 3) {
    Mat cell;
    vector<double> hist;

    int cnt = 0;
    block = lbpImg.rows/height;   // here we assume image is x by x and cell is y by y.
    
    for (int i = 0;i < block;i++)
    {
        for (int j = 0;j < block;j++)
        {
            cell = lbpImg(Rect(j*width, i*height, width, height)); //cout << cell.size() << endl;
            hist = lbp.calcHist(&cell, NULL).getHist(false);
            double * ptr = &hist[0];

            for (int k = 0; k<hist.size(); k++)  // 531 bn
            {
                histC.push_back(ptr[k]);
            }

        }
    }
}

static void makeFeaturesVector(Mat &imgOrg, LBP &lbp8, LBP &lbp16, vector<double> &histC, const int cellsize)
{
    vector<double> hist;
    double *ptr;
    Mat img;
    int rad = 2;
    int sizeH = 10;
    int sizeBlock = 3;

    Mat lbpImg;
    img = imgOrg;

    // LBP 8, 1
    lbp8.calcLBP(img, 1);
    lbpImg = lbp8.getLBPImage();
    spatial_Histogram(lbp8, lbpImg, cellsize, cellsize, histC);

    // LBP 8, 2
    lbp8.calcLBP(img, 2);
    lbpImg = lbp8.getLBPImage();
    spatial_Histogram(lbp8, lbpImg, cellsize, cellsize, histC);

    // LBP 16, 2
    lbp16.calcLBP(img, 2);
    lbpImg = lbp8.getLBPImage();
    spatial_Histogram(lbp8, lbpImg, cellsize, cellsize, histC);

    //***************************************//
    /*
    img = imgOrg;
    lbp8.calcLBP(img, rad);
    hist = lbp8.calcHist().getHist(false);
    ptr = &hist[0];
    for (int k = 0; k<hist.size(); k++)
    {
        histC.push_back(ptr[k]);
    }

    img = imgOrg;
    lbp16.calcLBP(img, rad);
    hist = lbp16.calcHist().getHist(false);
    ptr = &hist[0];

    for (int k = 0; k<hist.size(); k++)
    {
        histC.push_back(ptr[k]);
    }
    */
}

void getFaceLBPHist(cv::Mat &face, std::vector<double> &hist, const int cellsize)
{
    static LBP lbp8(8, LBP::strToType("riu2"));
    static LBP lbp16(16, LBP::strToType("riu2"));
    Mat tmp;
    cvtColor(face, tmp, CV_BGR2YCrCb);
    //cvtColor(face, tmp, CV_BGR2GRAY);

    hist.clear();
    for( int i = 0; i < tmp.channels(); i++ ) {
        Mat img( tmp.size(), tmp.depth(), 1);
        const int from_to1[] = { i, 0 };
        mixChannels( &tmp, 1, &img, 1, from_to1, 1);
        makeFeaturesVector(img, lbp8, lbp16, hist, cellsize);
    }
    double sum = 0;
    for (std::vector<double>::iterator it = hist.begin(); it != hist.end(); ++it) {
        sum += *it;
    }
    for (std::vector<double>::iterator it = hist.begin(); it != hist.end(); ++it) {
        *it = (*it) / sum;
    }
}