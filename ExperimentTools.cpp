#include "ExperimentTools.h"

void NaiveForwardNNWarp (Mat& input, Mat& flow, Mat& output, int ch)
{
	int new_i, new_j;

	if (ch == 1) {
		output = Mat::zeros(input.rows, input.cols, CV_8UC1);

		for (int i = 0; i < input.rows; i++) for (int j = 0; j < input.cols; j++) {
			Vec2f& tmp_flow = flow.at<Vec2f>(i,j);

			new_i = cvRound(i + tmp_flow.val[1]);
			new_j = cvRound(j + tmp_flow.val[0]);

			if (new_i >= input.rows || new_i < 0 || new_j >= input.cols || new_j < 0)
				continue;

			output.at<uchar>(new_i,new_j) = input.at<uchar>(i,j);
		}
	}
	else if (ch == 3) {
		output = Mat::zeros(input.rows, input.cols, CV_8UC3);

		for (int i = 0; i < input.rows; i++) for (int j = 0; j < input.cols; j++) {
			Vec2f& tmp_flow = flow.at<Vec2f>(i,j);

			new_i = cvRound(i + tmp_flow.val[1]);
			new_j = cvRound(j + tmp_flow.val[0]);

			if (new_i >= input.rows || new_i < 0 || new_j >= input.cols || new_j < 0)
				continue;

			Vec3b& target = output.at<Vec3b>(new_i,new_j);
			Vec3b& source = input.at<Vec3b>(i,j);
			target[0] = source[0];
			target[1] = source[1];
			target[2] = source[2];
		}
	}
	
}