#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "intrin.h"

using namespace cv;
using namespace std;

int main( )
{
	int ALFA = 0.5;
	// LOAD image
	cv::Mat A = cv::imread("A.png", IMREAD_GRAYSCALE);
	cv::Mat B = cv::imread("B.png", IMREAD_GRAYSCALE);
	unsigned char *A_data  = (unsigned char *) A.data;
	unsigned char *B_data  = (unsigned char *) B.data;
	int ANCOLS = A.cols;
	int ANROWS = A.rows;

	int BNCOLS = B.cols;
	int BNROWS = B.rows;

	cv::Mat img_ADD (ANROWS, ANCOLS, CV_8U);
	unsigned char *img_ADD_data = (unsigned char *) img_ADD.data;
	cv::Mat img_ADDp (ANROWS, ANCOLS, CV_8U);
	unsigned char *img_ADDp_data = (unsigned char *) img_ADDp.data;


	//SERIAL

	for (int row = 0; row < ANROWS; row++)
		for (int col = 0; col < ANCOLS; col++){
			if (*(B_data + row * BNCOLS + col) == 255)
				*(B_data + row * ANCOLS + col) = 0;
			*(img_ADD_data + row * ANCOLS + col) = (*(B_data + row * BNCOLS + col))*(0.5) + *(A_data + row * ANCOLS + col);
		}


	//PARALLEL
	__m128i *pSrcA, *pSrcB;
	__m128i *pRes;
	__m128i pa, pb, pAdd, m1, m2, m3, m4;

	pSrcA = (__m128i *) A.data;
	pSrcB = (__m128i *) B.data;
	pRes = (__m128i *) img_ADDp.data;

	m1 = _mm_set1_epi8((unsigned char) 0b11111110);
	m2 = _mm_set1_epi8((unsigned char) 0b11111111);

	for (int i = 0; i < ANROWS; i++)
		for (int j = 0; j < ANCOLS / 16; j++)
		{
			pa = _mm_loadu_si128(pSrcA + i * ANCOLS/16 + j) ;
			pb =_mm_loadu_si128(pSrcB + i * ANCOLS/16 + j) ;
			pb = _mm_cmplt_epi8(pb, m2);
			pb = _mm_and_si128(pb, m1);
			pb = _mm_srli_epi16(pb, 1);
			pAdd = _mm_add_epi8(pb, pa);
			_mm_storeu_si128 (pRes + i * ANCOLS/16 + j, pAdd);
		}

	//DISPLAY images
	cv::Mat show_A (ANROWS, ANCOLS, CV_8U); 
	cv::resize(A, show_A, cv::Size(), 1, 1);
	cv::namedWindow("A", cv::WINDOW_AUTOSIZE); 	
	cv::imshow("A", show_A);

	cv::Mat show_B (BNROWS, BNCOLS, CV_8U); 
	cv::resize(B, show_B, cv::Size(), 1, 1);
	cv::namedWindow("B", cv::WINDOW_AUTOSIZE); 	
	cv::imshow("B", show_B);

	cv::Mat show_img_ADD (img_ADD.rows, img_ADD.cols, CV_8U); 
	cv::resize(img_ADD, show_img_ADD, cv::Size(), 1, 1);
	cv::namedWindow("ADDs", cv::WINDOW_AUTOSIZE); 	
	cv::imshow("ADDs", show_img_ADD);

	cv::Mat show_img_ADDp (img_ADDp.rows, img_ADDp.cols, CV_8U); 
	cv::resize(img_ADDp, show_img_ADDp, cv::Size(), 1, 1);
	cv::namedWindow("ADDp", cv::WINDOW_AUTOSIZE); 	
	cv::imshow("ADDp", show_img_ADDp);

	waitKey(0);                       					// Wait for a keystroke in the window
	return 0;
}