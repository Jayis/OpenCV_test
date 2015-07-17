#include "GA_Reconstruction.h"

bool fitness_compare (Individual& a, Individual& b) { return (a.fitness < b.fitness); }

GA_Constructor::GA_Constructor( vector<Mat>& LR_imgs, vector<Mat>& flows, vector<Mat> confs, double scale, Mat& PSF) {
	cout << "----- GA-Constructor -----\n";

	int i, j, k;

	interp_scale = 512;

	LR_rows = LR_imgs[0].rows;
	LR_cols = LR_imgs[0].cols;
	HR_rows = LR_rows * scale;
	HR_cols = LR_cols * scale;
	LR_imgCount = LR_imgs.size();
	LR_pixelCount = LR_rows * LR_cols;
	HR_pixelCount = HR_rows * HR_cols;

	// pre-interpolation
	Mat super_PSF, super_BPk;
	preInterpolation ( PSF, super_PSF, interp_scale);
	super_BPk = super_PSF;

	//----- for every pixel x of HR image, record influenced pixel y
	// initialize bucket
	cout << "alloc HR_pix_array\n";
	HR_pixels = new HR_Pixel_Array(HR_rows, HR_cols);
	cout << "alloc LR_pix_array\n";
	// initialize influenced pixels (for each pixel in each LR img)
	LR_pixels = new LR_Pixel_Array(LR_imgCount, LR_rows, LR_cols);
	input_LR_imgs = LR_imgs;

	relations = new InfluenceRelation (LR_imgs,
							flows,
							LR_pixels,
							HR_pixels,
							scale,
							super_PSF,
							super_BPk,
							interp_scale);

	// GA parameters
	population_size = 1000000;
	generations = 100;
	srand (time(NULL));

	cout << "----- GA-Constructor ----- CONSTRUCT COMPLETE\n";
}

void GA_Constructor::solve()
{
	prepare();
	
	int cnt = 0;

	time_t time0, time1;
	time(&time0);

	Mat HRimg;

	cout << "start evolving...\n";
	while (cnt < generations)
	{
		cout << "*** Generation " << int2str(cnt) << " *** START!"<< endl;
		evolve();		

		time(&time1);
		cout << "*** Generation " << int2str(cnt) << " *** : " << difftime(time1, time0) << endl;\

		if (cnt%10 == 0) {
			output(HRimg);
			imwrite("output/generation" + int2str(cnt) + ".bmp", HRimg);
		}

		cout << "1st fitness: " << population[0].fitness << "\n";
		cout << "2st fitness: " << population[1].fitness << "\n";

		cnt++;
	}
}

void GA_Constructor::prepare()
{
	buffer.resize(population_size);

	population.resize(population_size);
	winners.resize(population_size/2);
	losers.resize(population_size/2);

	for (int i = 0; i < population_size; i++){
		buffer[i] = Mat::zeros(HR_rows, HR_cols, CV_8UC1);
	}

	init_individuals();	
}

void GA_Constructor::init_individuals()
{
	for (int i = 0; i < population_size; i++){
		population[i].image = buffer[i];
		randu(population[i].image, Scalar::all(0), Scalar::all(255));

		population[i].evaluated = false;
	}
}

void GA_Constructor::evolve()
{
	selection();
	crossover();
}

void GA_Constructor::ranking()
{
	cout << "ranking\n";
	evaluation();

	cout << "sort...\n";
	sort(population.begin(), population.end(), fitness_compare);
}

void GA_Constructor::evaluation()
{
	cout << "evaluation start\n";

	/*
	vector<Mat> tmp_LR = input_LR_imgs;
	Mat tmp_HR = imread("output/OptFlow_tv1/res256_LinearConstructC1_wConf.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	cout << tmp_HR.type() << endl;
	HRmat_to_LR_percetion ( tmp_HR, *LR_pixels, *relations);
	for (int k = 0; k < LR_imgCount; k++) {
		for (int i = 0; i < LR_rows; i++) for (int j = 0; j < LR_cols; j++) {
			tmp_LR[k].at<uchar>(i, j) = LR_pixels->access(k, i, j).perception;
		}
		imwrite("output/" + int2str(k) + ".bmp", tmp_LR[k]);
	}
	//*/

	for (int i = 0; i < population_size; i++) {
		Individual& individual = population[i];

		if (individual.evaluated) continue;

		individual.fitness = 0;
		
		HRmat_to_LR_percetion ( individual.image, *LR_pixels, *relations);
		
		for (int k = 0; k < LR_imgCount; k++) {
			for (int i = 0; i < LR_rows; i++) for (int j = 0; j < LR_cols; j++) {
				individual.fitness += abs(LR_pixels->access(k, i, j).perception - input_LR_imgs[k].at<uchar>(i, j)) / LR_pixelCount / LR_imgCount;
			}
		}
		/**/

		individual.evaluated = true;
	}
}

void GA_Constructor::selection()
{	
	cout << "selection start\n";
	ranking();

	for (int i = 0; i < population_size/2; i++) {
		winners[i] = &population[i];
		losers[i] = &population[population_size/2 + i];
	}
}

void GA_Constructor::crossover()
{
	cout << "crossover start\n";
	for (int i = 0; i < population_size/2; i+=2) {
		extendedLineXO(*winners[i], *winners[i+1], *losers[i], *losers[i+1]);
	}
}

void GA_Constructor::simpleXO(Individual& p1, Individual& p2, Individual& c1, Individual& c2)
{
	c1.evaluated = false;
	c2.evaluated = false;

	for (int i = 0; i < LR_rows; i++) for (int j = 0; j < LR_cols; j++)
	{
		if (rand()%2) {
			c1.image.at<uchar>(i, j) = p1.image.at<uchar>(i, j);
			c2.image.at<uchar>(i, j) = p2.image.at<uchar>(i, j);
		}
		else {
			c1.image.at<uchar>(i, j) = p2.image.at<uchar>(i, j);
			c2.image.at<uchar>(i, j) = p1.image.at<uchar>(i, j);
		}
	}
}

void GA_Constructor::extendedLineXO(Individual& p1, Individual& p2, Individual& c1, Individual& c2)
{
	c1.evaluated = false;
	c2.evaluated = false;
	
	/*
	double alpha = (double) rand() / RAND_MAX; 
	
	c1.image = alpha * p1.image + (1-alpha) * p2.image;
	c2.image = alpha * p2.image + (1-alpha) * p1.image;
	//*/

	double alpha;
	for (int i = 0; i < LR_rows; i++) for (int j = 0; j < LR_cols; j++)
	{
		alpha = (double) rand() / RAND_MAX * 1.5 - 0.25;

		c1.image.at<uchar>(i, j) = saturate_cast<uchar>(alpha * p1.image.at<uchar>(i, j) + (1-alpha) * p2.image.at<uchar>(i, j));
		c2.image.at<uchar>(i, j) = saturate_cast<uchar>(alpha * p2.image.at<uchar>(i, j) + (1-alpha) * p1.image.at<uchar>(i, j));
	}
}

void GA_Constructor::output(Mat& HRimg)
{
	population[0].image.copyTo(HRimg);
}

GA_Constructor::~GA_Constructor()
{
	delete HR_pixels;
	delete LR_pixels;
	delete relations;
}