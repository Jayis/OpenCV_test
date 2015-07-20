#pragma once

#include <opencv2\core\core.hpp>

#include "DataStructures.h"
#include "Methods.h"
#include "Macros.h"
#include "Tools.h"

using namespace cv;
using namespace Eigen;

class Linear_Constructor
{
public:
	Linear_Constructor( vector<Mat>& LR_imgs, vector<Mat>& flows, double scale, Mat& PSF);
	Linear_Constructor( vector<Mat>& LR_imgs, vector<Mat>& flows, vector<Mat> confs, double scale, Mat& PSF);
	~Linear_Constructor();
	
	void addRegularization_grad2norm(double gamma);

	void solve_bySparseQR();	// error
	void solve_byCG();

	void output(Mat& HRimg);

private:
	void constructor( vector<Mat>& LR_imgs, vector<Mat>& flows, vector<Mat> confs, double scale, Mat& PSF);

	void addDataFidelity( );
	void addDataFidelityWithConf(vector<Mat>& conf );

	vector<T> A_triplets;
	vector<double> b_vec;
	int curRow;
	VectorXd ATb, b, x;
	EigenSpMat A, ATA;

	int LR_imgCount;
	int LR_rows, LR_cols, HR_rows, HR_cols;
	int LR_pixelCount, HR_pixelCount;
	double interp_scale;

	HR_Pixel_Array* HR_pixels;
	LR_Pixel_Array* LR_pixels;
	InfluenceRelation* relations;
};
