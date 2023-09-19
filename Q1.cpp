#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "intrin.h"

using namespace cv;
using namespace std;

int main( )
{
	// LOAD image
	cv::Mat A = cv::imread("Q1/A.png", IMREAD_GRAYSCALE);
	cv::Mat B = cv::imread("Q1/B.png", IMREAD_GRAYSCALE);
	unsigned char *A_data  = (unsigned char *) A.data;
	unsigned char *B_data  = (unsigned char *) B.data;
	int NCOLS = A.cols;
	int NROWS = A.rows;

	// Convert to BW
	cv::Mat A_bw (NROWS, NCOLS, CV_8U);
	cv::Mat B_bw (NROWS, NCOLS, CV_8U);

	cv::Mat img_DIFF (NROWS, NCOLS, CV_8U);
	unsigned char *img_DIFF_data = (unsigned char *) img_DIFF.data;
	cv::Mat img_DIFF2 (NROWS, NCOLS, CV_8U);
	unsigned char *img_DIFF2_data = (unsigned char *) img_DIFF2.data;

	//SERIAL
	for (int row = 0; row < NROWS; row++)
		for (int col = 0; col < NCOLS; col++){
			*(img_DIFF_data + row * NCOLS + col) = 
			*(A_data + row * NCOLS + col) - *(B_data + row * NCOLS + col);
		}
	
	//PARALLEL
	__m128i *pSrcA, *pSrcB;
	__m128i *pRes;
	__m128i pa, pb, pSub;

	pSrcA = (__m128i *) A.data;
	pSrcB = (__m128i *) B.data;
	pRes = (__m128i *) img_DIFF2.data;

	for (int i = 0; i < NROWS; i++)
		for (int j = 0; j < NCOLS / 16; j++)
		{
			pa = _mm_loadu_si128(pSrcA + i * NCOLS/16 + j) ;
			pb =_mm_loadu_si128(pSrcB + i * NCOLS/16 + j) ;
			pSub = _mm_sub_epi8(pa, pb);
			_mm_storeu_si128 (pRes + i * NCOLS/16 + j, pSub);
		}


	//DISPLAY images
	cv::Mat show_A (NROWS, NCOLS, CV_8U); 
	cv::resize(A, show_A, cv::Size(), 1, 1);
	cv::namedWindow("A", cv::WINDOW_AUTOSIZE); 	
	cv::imshow("A", show_A);

	cv::Mat show_B (NROWS, NCOLS, CV_8U); 
	cv::resize(B, show_B, cv::Size(), 1, 1);
	cv::namedWindow("B", cv::WINDOW_AUTOSIZE); 	
	cv::imshow("B", show_B);

	cv::Mat show_img_DIFF (img_DIFF.rows/2, img_DIFF.cols/2, CV_8U); 
	cv::resize(img_DIFF, show_img_DIFF, cv::Size(),1, 1);
	cv::namedWindow("DIFF", cv::WINDOW_AUTOSIZE); 	
	cv::imshow("DIFF", show_img_DIFF);

	cv::Mat show_img_DIFF2 (img_DIFF2.rows/2, img_DIFF2.cols/2, CV_8U); 
	cv::resize(img_DIFF2, show_img_DIFF2, cv::Size(),1, 1);
	cv::namedWindow("DIFF p", cv::WINDOW_AUTOSIZE); 	
	cv::imshow("DIFF p", show_img_DIFF2);

	waitKey(0);                       					// Wait for a keystroke in the window
	return 0;
}