#pragma once

#include <opencv2\core\core.hpp>

#include <vector>
#include <iostream>

//#include "Linear_Reconstruction.h"
#include "Macros.h"

using namespace std;
using namespace cv;

class Pixel {
public:
	Pixel();

	double pos_x, pos_y;
	double val;
};

class Influenced_Pixel;
class Perception_Pixel;
class DataChunk;

class HR_Pixel : public Pixel {
public:
	HR_Pixel();

	int i, j;
	double hBP_sum, var;
	//vector<Influenced_Pixel> influenced_pixels;
	int influence_link_start;
	int influence_link_cnt;

	int highVar_idx;
};

class LR_Pixel : public Pixel {
public:
	LR_Pixel();

	int i, j, k;

	double confidence;
	double perception;

	int perception_link_start;
	int perception_link_cnt;
};

class Influenced_Pixel {
public:
	double hBP;
	LR_Pixel* pixel;
	int lr_idx;
	int hr_idx;
};

class Perception_Pixel {
public:
	double hPSF;
	HR_Pixel* pixel;
	int lr_idx;
	int hr_idx;
};

class HR_Pixel_Array {
public:
	HR_Pixel_Array(int r, int c);
	~HR_Pixel_Array();

	HR_Pixel& access(int i, int j);
	HR_Pixel& access(int idx);

	void clear_link();

	int HR_rows, HR_cols;
	int HR_pixelCount;
private:
	HR_Pixel* hr_pixels;
};

class LR_Pixel_Array {
public:
	LR_Pixel_Array(int k, int r, int c);
	~LR_Pixel_Array();

	LR_Pixel& access(int k, int i, int j);
	LR_Pixel& access(int idx);

	void clear_link();

	int LR_imgCount;
	int LR_pixelCount;
	int LR_rows, LR_cols;
private:
	LR_Pixel* lr_pixels;
};

class InfluenceRelation {
public:
	InfluenceRelation(vector<Mat>& imgs,
		vector<Mat>& flows,
		LR_Pixel_Array* LR_pixels,
		HR_Pixel_Array*  HR_pixels,
		double scale,
		Mat& super_PSF,
		Mat& super_BPk,
		double interp_scale, 
		vector<Mat>& confs);
	InfluenceRelation(vector<Mat>& imgs,
		vector<Mat>& flows,
		LR_Pixel_Array* LR_pixels,
		HR_Pixel_Array*  HR_pixels,
		double scale,
		Mat& super_PSF,
		Mat& super_BPk,
		double interp_scale);
	InfluenceRelation(
		DataChunk& dataChunk,
		Mat& super_PSF,
		Mat& super_BPk,
		double interp_scale);

	//~InfluenceRelation();

	vector<Influenced_Pixel> influence_links;
	vector<Perception_Pixel> perception_links;

private:
	void constructor(vector<Mat>& imgs,
							vector<Mat>& flows,
							LR_Pixel_Array* LR_pixels,
							HR_Pixel_Array*  HR_pixels,
							double scale,
							Mat& super_PSF,
							Mat& super_BPk,
							double interp_scale,
							vector<Mat>& confs);
};

class DataChunk
{
public:
	Mat smallHR;
	Rect inBigHR, inSmallHR;
	vector<LR_Pixel> data_LR_pix;

	HR_Pixel_Array*  tmp_HR_pixels;
	//LR_Pixel_Array* LR_pixels;

	int leftBorder, rightBorder, upBorder, downBorder;
	int SmallHR_rows, SmallHR_cols;
	int blockRowIdx, blockColIdx;
	int overLappingPix;

	InfluenceRelation* tmp_relations;

	int highVar_cnt;
	bool fullReconstruct;
};

//-----SparseMat
class Element {
public:
	Element(int ii, int jj, double value);

	int i, j;
	double val;
};

class MySparseMat {
public:
	MySparseMat();
	MySparseMat(int r, int c, int t);

	void insertElement(Element& e);
	void setVal(int i, int j, double val);
	double getVal(int i, int j);

	int type; // row_major = 0, col_major = 1
	int rows, cols;
	int nzcount;

	vector<vector<Element> > elements;
};