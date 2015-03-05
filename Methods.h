#pragma once

#include <iostream>

#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <Eigen\Sparse>

#include "DataStructures.h"
#include "Macros.h"

using namespace cv;
using namespace std;

// BP
void formInfluenceRelation (vector<Mat>& imgs,
							vector<Mat>& flows,
							vector < vector < vector <LR_Pixel> > >& LR_pixels,
							vector < vector <HR_Pixel> >&  influence_bucket,
							double scale,
							Mat& super_PSF,
							Mat& super_BPk,
							double interp_scale);

void HR_to_LR ( Mat& HRimg, Mat& LRimg, double scale, Mat& PSF, bool super_PSF, double PSF_scale=0 );

void preInterpolation ( Mat& PSF, Mat& super_PSF, double PSF_scale );

void HR_to_LR_percetion ( Mat& HRimg, vector < vector < vector <LR_Pixel> > >& LR_pixels, double scale, Mat& PSF, bool is_super_PSF, double PSF_scale=0 );
// OptFlow
void NaiveForwardNNWarp (Mat& source, Mat& flow, Mat& output, int ch);

void showConfidence (Mat& flow_forward, Mat& flow_backward, Mat& confidence);

double ExpNegSQR (float x);

double calcConfidence (Vec2f& f, Vec2f& b);
// FlexISP
void resampleByMatrix (Mat& X,
					   vector <EigenSpMat>& S, 
					   vector <Mat>& SX,
					   int LR_Rows,
					   int LR_Cols);

void formSparseI (EigenSpMat& out, int rows, int cols);