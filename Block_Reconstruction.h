#pragma once

#include <opencv2\core\core.hpp>

#include "DataStructures.h"
#include "Methods.h"

using namespace cv;

class Block_Constructor
{
public:
	Block_Constructor(vector<Mat>& imgs,
									 vector<Mat>& flows,
									 vector<Mat>& confs,
									 double scale,
									 Mat& PSF);

	vector< vector< DataChunk > > dataChunks;

	void construct(Mat& super_PSF,
		Mat& super_BPk,
		double interp_scale);

	void output(Mat& HRimg);

private:
	int overlappingPix;
	int BigLR_rows, BigLR_cols, BigHR_rows, BigHR_cols;
	int LR_imgCount;
	int longSide, totalBlocksCount;
	double blockPerAxis, blockWidth, blockHeight;

	//HR_Pixel_Array* HR_pixels;
	LR_Pixel_Array* LR_pixels;
};