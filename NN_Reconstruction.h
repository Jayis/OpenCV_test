#pragma once

#include <algorithm>

#include <opencv2\core\core.hpp>

#include "DataStructures.h"
#include "Methods.h"

using namespace cv;

class NN_Constructor
{
public:
	NN_Constructor( vector<Mat>& LR_imgs, vector<Mat>& flows, vector<Mat> confs, double scale, Mat& PSF);
	NN_Constructor( DataChunk& dataChunk );
	~NN_Constructor();

	void solve();
	void solve_by_LinearRefine(DataChunk& dataChunk);

	void output(Mat& HRimg);

private:
	void constructor( vector<Mat>& LR_imgs, vector<Mat>& flows, vector<Mat> confs, double scale, Mat& PSF);

	int K;

	int LR_imgCount;
	int LR_rows, LR_cols, HR_rows, HR_cols;
	int LR_pixelCount, HR_pixelCount;
	double interp_scale;

	int rim;

	HR_Pixel_Array* HR_pixels;
	LR_Pixel_Array* LR_pixels;
	InfluenceRelation* relations;

	bool needRelease;
};

bool compare_hpsf (Influenced_Pixel* a, Influenced_Pixel* b);