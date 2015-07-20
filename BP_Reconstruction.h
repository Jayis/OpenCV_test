#pragma once

#include <opencv2\core\core.hpp>

#include "DataStructures.h"
#include "Methods.h"
#include "Macros.h"
#include "Tools.h"

using namespace cv;

class BP_Constructor
{
public:
	BP_Constructor( Mat& HRimg, double scale, vector<Mat>& imgs, vector<Mat>& flows, Mat& PSF, Mat& BPkernel, TermCriteria& BPstop );
	BP_Constructor( Mat& HRimg, double scale, vector<Mat>& imgs, vector<Mat>& flows, Mat& PSF, Mat& BPkernel, TermCriteria& BPstop,  vector<Mat>& confidences );
	~BP_Constructor();

	void output(Mat& HRimg);

private:
	void constructor( Mat& HRimg, double scale, vector<Mat>& imgs, vector<Mat>& flows, Mat& PSF, Mat& BPkernel, TermCriteria& BPstop, vector<Mat>& confidences );
	void setHRinitial();
	void solve();

	double BP_c;
	double interp_scale;
	TermCriteria BPtermination;

	int LR_rows;
	int LR_cols;
	int HR_rows;
	int HR_cols;
	int LR_imgCount;

	HR_Pixel_Array* HR_pixels;
	LR_Pixel_Array* LR_pixels;
	InfluenceRelation* relations;
};