#pragma once

#include <opencv2\core\core.hpp>

#include <vector>
#include <iostream>

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

class HR_Pixel : public Pixel {
public:
	HR_Pixel();

	int i, j;
	double hBP_sum;
	//vector<Influenced_Pixel> influenced_pixels;
	int influence_link_start;
	int influence_link_cnt;
};

class LR_Pixel : public Pixel {
public:
	LR_Pixel();

	int i, j, k;

	double confidence;
	double perception;
	//vector<Perception_Pixel> perception_pixels;
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