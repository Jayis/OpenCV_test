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