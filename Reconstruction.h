#pragma once

#include "DataStructures.h"
/*
class Constructor
{
public:
	Constructor( vector<Mat>& LR_imgs, vector<Mat>& flows, double scale, Mat& PSF);
	Constructor( vector<Mat>& LR_imgs, vector<Mat>& flows, vector<Mat> confs, double scale, Mat& PSF);

	Constructor();


private:
	void constructor();
	void constructor( vector<Mat>& LR_imgs, vector<Mat>& flows, vector<Mat> confs, double scale, Mat& PSF);

	void set_variables();
	void alloc_construction_mem();

	int LR_imgCount;
	int LR_rows, LR_cols, HR_rows, HR_cols;
	int LR_pixelCount, HR_pixelCount;
	double interp_scale;

	HR_Pixel_Array* HR_pixels;
	LR_Pixel_Array* LR_pixels;

	vector<HR_Pixel*> HR_pixel_vec;
	vector<LR_Pixel*> LR_pixel_vec;

	InfluenceRelation* relations;
};
//*/