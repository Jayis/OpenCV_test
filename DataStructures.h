#pragma once

#include <vector>

using namespace std;

class Pixel {
public:
	Pixel();

	double pos_x, pos_y;
	double val;
	double confidence;
	double percetion;
	int i, j, k;
};

class Influenced_Pixel {
public:
	double hBP;
	Pixel* pixel;
};

class Influence_Bucket {
public:
	double hBP_sum;
	vector<Influenced_Pixel> influenced_pixels;
};