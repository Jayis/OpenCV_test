#include <iostream>
#include <opencv2\core\core.hpp>

using namespace std;
using namespace cv;

void estimate_H ( vector<Mat>& output, vector<Mat>& input, int H_array_rows, int H_array_cols, double lambda ) {
	int input_size = input.size();
	
	vector <vector <int> > connection;
	vector <Mat> tmp[2];
	
	output.resize(input_size);
	connection.resize(input_size);
	tmp[0].resize(input_size);
	tmp[1].resize(input_size);

	int H_r = input[0].rows;
	int H_c = input[0].cols;

	for (int i=0; i<output.size(); i++) {
		output[i] = Mat::zeros( H_r, H_c, CV_32F);
		tmp[0][i] = Mat::ones( H_r, H_c, CV_32F);
		tmp[1][i] = Mat::ones( H_r, H_c, CV_32F);
	}
	// construct (A'A)x = A'b  and A'A will be a square matrix
	
	int index = 0;
	for (int i=0; i < H_array_rows; i++) {
		for (int j=0; j < H_array_cols; j++) {
			index = i*H_array_cols + j;

			if (i > 0)
				connection[index].push_back(index-H_array_cols);
			if (i < H_array_rows - 1)
				connection[index].push_back(index+H_array_cols);
			if (j > 0)
				connection[index].push_back(index-1);
			if (j < H_array_cols - 1)
				connection[index].push_back(index+1);
		}
	}
	/**/
	// true = 1, false = 0
	bool turn = false;
	double diff = 1e8;
	int loop = 0;
	while (diff > 0.001) {
		/*
		cout <<"loop: " << loop++ << endl;
		cout << "difference: " << diff << endl;
		/**/
		diff = 0;
		
		vector <Mat>& tmp_H_old = tmp[turn];
		turn = !turn;
		vector <Mat>& tmp_H_new = tmp[turn];

		// ith H
		for (int i=0; i<input.size(); i++) {
			// (1+coneection_num*lambda)*H_new = R + lambda*(sigma_connect  H_old)
			input[i].copyTo(tmp_H_new[i]);
			for (int k=0; k<connection[i].size(); k++) {
				tmp_H_new[i] += lambda*tmp_H_old[connection[i][k]];
			}
			tmp_H_new[i] /= (1 + connection[i].size()*lambda);

			for (int r=0; r < H_r; r++) {
				for (int c=0; c < H_c; c++) {
					diff += abs( tmp_H_new[i].at<float>(r,c) - tmp_H_old[i].at<float>(r,c) );
				}
			}
		}
		diff /= (input.size() * H_r * H_c);
		/*
		for (int i = 0; i < input.size(); i++) {
			cout << input[i] << endl;
			cout << tmp_H_new[i] << endl;
			cout << tmp_H_old[i] << endl;
		}
		/**/
	}
	cout << "convergence difference: " << diff << endl;

	for (int i = 0; i < input.size(); i++)
		output[i] = tmp[turn][i];
}

int main () {

	vector <Mat> in;
	vector <Mat> out;

	Mat aa = Mat::eye(3,3,CV_32F);
	Mat bb = Mat::ones(3,3,CV_32F);

	for (int i = 0; i < 3; i++) {
		for (int j= 0; j < 3; j++) {
			aa.at<float>(i,j) = i*3+j;
		}
	}

	for (int i = 0; i < 6; i++) {
		in.push_back(i*aa);
	}

	estimate_H(out, in, 2, 3, 0.01);
	
	for (int i = 0; i < in.size(); i++)
		cout << in[i] << endl;
	
	for (int i = 0; i < out.size(); i++)
		cout << out[i] << endl;
	/*	*/
	system("pause");
	return 0;
}