#pragma once

#include <time.h>

#include <opencv2\core\core.hpp>

#include "Methods.h"

using namespace cv;

class Individual
{
public:
	bool evaluated;
	double fitness;
	Mat image;

	/*
	void operator = (Individual& a) {
		evaluated = a.evaluated; fitness = a.fitness; image = a.image;
	}
	//*/
};

class GA_Constructor
{
public:
	GA_Constructor( vector<Mat>& LR_imgs, vector<Mat>& flows, double scale, Mat& PSF);
	GA_Constructor( vector<Mat>& LR_imgs, vector<Mat>& flows, vector<Mat> confs, double scale, Mat& PSF);
	~GA_Constructor();

	void solve();

	void output(Mat& HRimg);

private:
	void prepare();
	void init_individuals();
	void evolve();
	void evaluation();
	void ranking();
	void selection();
	void crossover();

	void pairwiseXO(Individual& p1, Individual& p2, Individual& c1, Individual& c2);
	void simpleXO(Individual& p1, Individual& p2, Individual& c1, Individual& c2);
	void extendedLineXO(Individual& p1, Individual& p2, Individual& c1, Individual& c2);

	int LR_imgCount;
	int LR_rows, LR_cols, HR_rows, HR_cols;
	int LR_pixelCount, HR_pixelCount;
	double interp_scale;

	HR_Pixel_Array* HR_pixels;
	LR_Pixel_Array* LR_pixels;
	InfluenceRelation* relations;

	vector<Mat> input_LR_imgs;

	int population_size, generations;
	vector<Individual> population;	

	vector<Mat> buffer;

	vector<Individual*> winners, losers;
};