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
	Linear_Constructor( DataChunk& dataChunk );
	~Linear_Constructor();
	
	void addRegularization_grad2norm(double gamma);

	void solve_by_CG();
	void solve_by_GradientDescent();

	void output(Mat& HRimg);

	bool needRelease;

private:
	void constructor( vector<Mat>& LR_imgs, vector<Mat>& flows, vector<Mat> confs, double scale, Mat& PSF);

	void addDataFidelityWithConf(vector<Mat>& conf);
	void addDataFidelityWithConf(DataChunk& dataChunk);

	vector<T> A_triplets, C_triplets;
	vector<double> b_vec;
	int rowCnt_A, rowCnt_C;
	VectorXd ATb, b, x, x_n, x_n1;
	EigenSpMat A, ATA;
	EigenSpMat C, CTC;

	int LR_imgCount;
	int LR_rows, LR_cols, HR_rows, HR_cols;
	int LR_pixelCount, HR_pixelCount;
	double interp_scale;

	HR_Pixel_Array* HR_pixels;
	LR_Pixel_Array* LR_pixels;
	InfluenceRelation* relations;
};
