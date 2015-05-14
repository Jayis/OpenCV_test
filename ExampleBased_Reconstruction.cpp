#include "ExampleBased_Reconstruction.h"

void mySearch (Mat& output, Mat& dataImg, Mat& queryImg, Rect& queryRect, int* HR_exist, double hr_weight)
{
	int dataImg_rows = dataImg.rows, dataImg_cols = dataImg.cols;
	int query_w = queryRect.width, query_h = queryRect.height;
	int query_x = queryRect.x, query_y = queryRect.y;

	Mat dot = Mat::zeros(5,5,CV_64F);
	dot.at<double>(2, 2) = 1; 
	Mat PSF = Mat::zeros(5,5,CV_64F);
	GaussianBlur(dot, PSF, Size( 5, 5), 1, 1, BORDER_REPLICATE);
	double scale = 2;
	double interpScale = 1000;
	Mat superPSF;
	preInterpolation(PSF, superPSF, interpScale);

	Mat queryLR, tmp;
		//imwrite("output/queryHR.bmp", queryImg(queryRect));
	HR_to_LR ( queryImg(queryRect), queryLR, scale, superPSF, true, interpScale );
	resize(queryLR, tmp, Size(0, 0), 300, 300);
	imwrite("output/qLR1.bmp", tmp);
	/*
	HR_to_LR ( queryImg(queryRect), queryLR, scale, PSF, false);
	resize(queryLR, tmp, Size(0, 0), 300, 300);
	imwrite("output/qLR2.bmp", tmp);
	*/
	double totalDist = 1e8;
	int totalSearch = (dataImg_rows - 2*query_h) * (dataImg_cols - 2*query_w);
	
	Mat LR_dist = Mat::zeros(dataImg_rows, dataImg_cols, CV_64F);
	Mat HR_up_dist = Mat::zeros(dataImg_rows, dataImg_cols, CV_64F);
	Mat HR_down_dist = Mat::zeros(dataImg_rows, dataImg_cols, CV_64F);
	Mat HR_left_dist = Mat::zeros(dataImg_rows, dataImg_cols, CV_64F);
	Mat HR_right_dist = Mat::zeros(dataImg_rows, dataImg_cols, CV_64F);

	int idx = 0;
	// LR
	
	for (int i = query_h; i < dataImg_rows-query_h; i++) for (int j = query_w; j < dataImg_cols-query_w; j++) {		
		cout << "\rsearchingLR..." << (double)idx/totalSearch;
		idx++;
		
		Mat dataLR;
		//imwrite("output/dataHR.bmp", dataImg(Rect(j, i, query_w, query_h)));
		HR_to_LR ( dataImg(Rect(j, i, query_w, query_h)), dataLR, scale, superPSF, true, interpScale );
		resize(dataLR, tmp, Size(0, 0), 300, 300);
		//imwrite("output/dLR1.bmp", tmp);

		double dist = 0;
		for (int ii = 0; ii < queryLR.rows; ii++) for (int jj = 0; jj <queryLR.cols; jj++) {
			dist += abs(dataLR.at<double>(ii, jj) - queryLR.at<double>(ii, jj));
		}
		
		LR_dist.at<double>(i, j) = dist;
	}
	/**/
	// HR
		idx = 0;
		if (HR_exist[0]!=0) {// up
			Mat query = queryImg(Rect(query_x, query_y - query_h, query_w, query_h));

			for (int i = query_h; i < dataImg_rows-query_h; i++) for (int j = query_w; j < dataImg_cols-query_w; j++) {
				
				cout << "\rsearchingHRup..." << (double)idx/totalSearch;
				idx++;

				Mat data = dataImg(Rect(j, i - query_h, query_w, query_h));				
				double dist = 0;
				for (int ii = 0; ii < query.rows; ii++) for (int jj = 0; jj <query.cols; jj++) {
					dist += abs(data.at<double>(ii, jj) - query.at<double>(ii, jj));
				}
				
				HR_up_dist.at<double>(i,j) = dist;
			}			
		}

		idx = 0;
		if (HR_exist[1]!=0) {// down
			Mat query = queryImg(Rect(query_x, query_y - query_h, query_w, query_h));

			for (int i = query_h; i < dataImg_rows-query_h; i++) for (int j = query_w; j < dataImg_cols-query_w; j++) {
				cout << "\rsearchingHRdown..." << (double)idx/totalSearch;
				idx++;

				Mat data = dataImg(Rect(j, i - query_h, query_w, query_h));				
				double dist = 0;
				for (int ii = 0; ii < query.rows; ii++) for (int jj = 0; jj <query.cols; jj++) {
					dist += abs(data.at<double>(ii, jj) - query.at<double>(ii, jj));
				}
				
				HR_down_dist.at<double>(i,j) = dist;
			}			
		}

		idx = 0;
		if (HR_exist[2]!=0) {// left
			Mat query = queryImg(Rect(query_x, query_y - query_h, query_w, query_h));

			for (int i = query_h; i < dataImg_rows-query_h; i++) for (int j = query_w; j < dataImg_cols-query_w; j++) {
				cout << "\rsearchingHRleft..." << (double)idx/totalSearch;
				idx++;

				Mat data = dataImg(Rect(j, i - query_h, query_w, query_h));				
				double dist = 0;
				for (int ii = 0; ii < query.rows; ii++) for (int jj = 0; jj <query.cols; jj++) {
					dist += abs(data.at<double>(ii, jj) - query.at<double>(ii, jj));
				}
				
				HR_left_dist.at<double>(i,j) = dist;
			}			
		}

		idx = 0;
		if (HR_exist[3]!=0) {// right
			Mat query = queryImg(Rect(query_x, query_y - query_h, query_w, query_h));
			for (int i = query_h; i < dataImg_rows-query_h; i++) for (int j = query_w; j < dataImg_cols-query_w; j++) {
				cout << "\rsearchingHRright..." << (double)idx/totalSearch;
				idx++;

				Mat data = dataImg(Rect(j, i - query_h, query_w, query_h));				
				double dist = 0;
				for (int ii = 0; ii < query.rows; ii++) for (int jj = 0; jj <query.cols; jj++) {
					dist += abs(data.at<double>(ii, jj) - query.at<double>(ii, jj));
				}
				
				HR_right_dist.at<double>(i,j) = dist;
			}			
		}

	for (int i = query_h; i < dataImg_rows-query_h; i++) for (int j = query_w; j < dataImg_cols-query_w; j++) {
		int idx = 0;
		cout << "\rfinding min..." << (double)idx/totalSearch;
		idx++;

		double HR_total_dist = HR_up_dist.at<double>(i, j) + HR_down_dist.at<double>(i, j) + HR_left_dist.at<double>(i, j) + HR_right_dist.at<double>(i, j);

		if ((hr_weight * HR_total_dist + (1-hr_weight) * LR_dist.at<double>(i, j)) < totalDist) {
			totalDist = (hr_weight * HR_total_dist + (1-hr_weight) * LR_dist.at<double>(i, j));
			output = dataImg(Rect(j, i, query_w, query_h));
		}

	}

	
		
}