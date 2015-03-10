#pragma once

#include <iostream>

#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\photo\photo.hpp>
#include <opencv2\highgui\highgui.hpp>

#include <Eigen\Core>

#include "Methods.h"
#include "Macros.h"
#include "Tools.h"

using namespace std;
using namespace cv;
using namespace Eigen;

void FlexISPmain (vector<Mat>& imgsC1, vector<Mat>& flows, vector<Mat>& confs, Mat& PSF, Mat& BPk, double scale, Mat& output);

void FirstOrderPrimalDual (double gamma, double tau, double theta, Mat& x_0, Mat& tauATz, EigenSpMat& tauATAplusI, Mat& result);

void penalty (vector<Mat>& y, Mat& x_bar, double gamma);

void data_fidelity (Mat& x_k1, Mat& x_k, vector<Mat>& y, double tau, Mat& tauATz, ConjugateGradient<EigenSpMat>& cg);

void extrapolation (Mat& x_bar, Mat& x_k1, Mat& x_k, double theta);

void formResampleMatrix (vector < vector < vector <LR_Pixel> > >& LR_pixels,
							  vector < vector <HR_Pixel> >&  HR_pixels,
							  vector <MySparseMat>& S,
							  vector <MySparseMat>& ST,
							  vector <EigenSpMat>& ST_eigen);

void form_tauATAplusI (double tau, vector<MySparseMat>& ST, vector<Mat>& conf, vector<MySparseMat>& S, EigenSpMat& out);

// doesn't include v
void form_tauATz (double tau, vector<MySparseMat>& ST, vector<Mat>& conf, vector<Mat>& LRimgs, Mat& out, int HR_rows, int HR_cols);
void form_tauATz_eigen (double tau, vector<EigenSpMat>& ST, vector<Mat>& conf, vector<Mat>& LRimgs, Mat& out, int HR_rows, int HR_cols);