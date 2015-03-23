#pragma once

#include <vector>

using namespace std;

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
	vector<Influenced_Pixel> influenced_pixels;
};

class LR_Pixel : public Pixel {
public:
	LR_Pixel();

	int i, j, k;

	double confidence;
	double perception;
	vector<Perception_Pixel> perception_pixels;
};

class Influenced_Pixel {
public:
	double hBP;
	LR_Pixel* pixel;
};

class Perception_Pixel {
public:
	double hPSF;
	HR_Pixel* pixel;
};

class HR_Pixel_Array {
public:
	HR_Pixel_Array(int r, int c);
	~HR_Pixel_Array();

	HR_Pixel& access(int i, int j);

private:
	int HR_rows, HR_cols;
	int HR_pixelCount;
	HR_Pixel* hr_pixels;
};

class LR_Pixel_Array {
public:
	LR_Pixel_Array(int k, int r, int c);
	~LR_Pixel_Array();

	LR_Pixel& access(int k, int i, int j);

private:
	int LR_imgCount;
	int LR_pixelCount;
	int LR_rows, LR_cols;
	LR_Pixel* lr_pixels;
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

