#pragma once

#include <opencv2\core\core.hpp>

#include "DataStructures.h"
#include "Methods.h"
#include "Macros.h"

using namespace cv;
using namespace Eigen;

class LinearConstructor
{
public:
	LinearConstructor( vector<Mat>& LR_imgs, vector<Mat>& flows, double scale, Mat& PSF);
	
	void addRegularization_grad2norm();

	void solve_bySparseQR();	// error
	void solve_byCG();

	void output(Mat& HRimg);

private:
	void addDataFidelityTerm( );

	vector<T> A_triplets;
	vector<double> b_vec;
	int curRow;
	VectorXd ATb, b, x;
	EigenSpMat A, ATA;

	int LR_imgCount;
	int LR_rows, LR_cols, HR_rows, HR_cols;
	int LR_pixelCount, HR_pixelCount;
	double interp_scale;
	vector < vector <HR_Pixel> >  HR_pixels;
	vector < vector < vector <LR_Pixel> > > LR_pixels;
};