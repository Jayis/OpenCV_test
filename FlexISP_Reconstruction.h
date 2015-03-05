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

void FirstOrderPrimalDual ();

void penalty (vector<Mat>& y, Mat& x_bar, double gamma);

void data_fidelity (Mat& x_k1, Mat& x_k, vector<Mat>& y, double tau);

void extrapolation (Mat& x_bar, Mat& x_k1, Mat& x_k, double theta);

void form_tauATAplusI (double tau, vector<EigenSpMat>& ST, vector<Mat>& conf, vector<EigenSpMat>& S);

// doesn't include v
void form_tauATz (double tau, vector<EigenSpMat>& ST, vector<Mat>& conf, vector<Mat>& LRimgs, Mat& out, int HR_rows, int HR_cols) ;