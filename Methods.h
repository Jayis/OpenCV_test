#pragma once

#include <iostream>

#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <Eigen\Sparse>
#include <opencv2\legacy\legacy.hpp>

#include "DataStructures.h"
#include "Linear_Reconstruction.h"
#include "Macros.h"

using namespace cv;
using namespace std;

// BP
void formInfluenceRelation (vector<Mat>& imgs,
							vector<Mat>& flows,
							LR_Pixel_Array* LR_pixels,
							HR_Pixel_Array* HR_pixels,
							double scale,
							Mat& super_PSF,
							Mat& super_BPk,
							double interp_scale);

void HR_to_LR ( Mat& HRimg, Mat& LRimg, double scale, Mat& PSF, bool super_PSF, double PSF_scale=0 );

void preInterpolation ( Mat& PSF, Mat& super_PSF, double PSF_scale );

void HR_to_LR_percetion ( Mat& HRimg, LR_Pixel_Array& LR_pixels, InfluenceRelation& relations, Mat& PSF, bool is_super_PSF, double PSF_scale=0 );
// OptFlow
void NaiveForwardNNWarp (Mat& source, Mat& flow, Mat& output, int ch);

void showConfidence_new (Mat& flow_forward, Mat& flow_backward, Mat& confidence);

void showConfidence (Mat& flow_forward, Mat& flow_backward, Mat& confidence, double sigmaScaler = 1);

double ExpNegSQR (double x, double y, double sigma = 0.4343);

double calcConfidence (Vec2f& f, Vec2f& b, double sigma = 0.4343);

void ImgPreProcess (vector<Mat>& LR_imgs, vector<Mat>& output);

void getBetterFlow (vector<Mat>& oriConfs, vector<Mat>& OriFlow, vector<Mat>& newConfs, vector<Mat>& newFlows, vector<Mat>& combinedConfs, vector<Mat>& combinedFlows);
// FlexISP
void formResampleMatrix (vector < vector < vector <LR_Pixel> > >& LR_pixels,
							  vector < vector <HR_Pixel> >&  HR_pixels,
							  vector <MySparseMat>& S,
							  vector <MySparseMat>& ST,
							  vector <EigenSpMat>& S_eigen,
							  vector <EigenSpMat>& ST_eigen);

void resampleByMatrix (Mat& X,
					   vector <MySparseMat>& S, 
					   vector <Mat>& SX,
					   int LR_Rows,
					   int LR_Cols);

void formSparseI (EigenSpMat& out, int rows, int cols);

void multiplyMySpMat (MySparseMat& A, MySparseMat& B, EigenSpMat& out);

double MySpMat_dot (vector<Element>& a, vector<Element>& b);

void DivideToBlocksToConstruct(vector<Mat>& BigLRimgs, vector<Mat>& BigFlows, vector<Mat>& BigConfs, Mat& PSF, double scale, Mat& BigHRimg);
// 
void weightedNeighborWarp (vector<vector<HR_Pixel> >& HR_pixels, Mat& HRimg);

void warpImageByFlow (Mat& colorImg, Mat& flow, Mat& output);

void outputHRcolor (Mat& HRimgC1, Mat& LRimg, Mat& HRimgC3);

void specialBlur (Mat& input, Mat& output);
// 
void optFlowHS (Mat& prev, Mat& curr, Mat& flow);