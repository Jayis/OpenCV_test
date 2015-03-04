#include "FlexISP_Reconstruction.h"

void penalty (vector<Mat>& y, Mat& x_bar, double gamma) {
	// 
	double inv_gamma = 1.0 / gamma;

	//---- v = y + gamma * K * x_bar
	vector<Mat> v;
	v.resize(3);
	for (int k = 0; k < 3; k++) {
		v[k] = Mat::zeros(x_bar.rows, x_bar.cols, CV_64F);
	}
	
	Mat grad_x, grad_y;
	Sobel( x_bar, grad_x, CV_64F, 1, 0, 3, 1, 0, BORDER_DEFAULT );
	Sobel( x_bar, grad_y, CV_64F, 0, 1, 3, 1, 0, BORDER_DEFAULT );

	v[0] = y[0] + (gamma * grad_x);
	v[1] = y[1] + (gamma * grad_y);
	v[2] = y[2] + (gamma * x_bar);

	for (int k = 0; k < 3; k++) {
		imwrite ("Flex_output/v_" + int2str(k) + ".png", v[k]);
	}

	// calculate proximal-inv_gamma-F(v/gamma)
	// first proximal operator
	// TV 1, image gradient sparsity prior
	vector<Mat> proxF;
	proxF.resize(3);
	for (int k = 0; k < 3; k++) {
		proxF[k] = Mat::zeros(x_bar.rows, x_bar.cols, CV_64F);
	}

	for (int k = 0; k < 2; k++) {
		for (int i = 0; i < v[k].rows; i++) for (int j = 0; j < v[k].cols; j++) {
			double& cur_element = proxF[k].at<double>(i, j);

			cur_element = v[k].at<double>(i, j) * inv_gamma;

			if ( cur_element > inv_gamma ) {
				cur_element -= inv_gamma;
			}
			else if ( cur_element < inv_gamma ) {
				cur_element += inv_gamma;
			}
			else {
				cur_element = 0;
			}
		}
	}
	// second proximal operator
	// Denoise
	Mat tmp_proxF2, tmp_proxF2_8bit, tmp_proxF2_denoise;
	tmp_proxF2 = Mat::zeros(v[2].rows, v[2].cols, CV_64F);
	tmp_proxF2 = v[2] * inv_gamma;
	tmp_proxF2.convertTo( tmp_proxF2_8bit, CV_8U );
	fastNlMeansDenoising( tmp_proxF2_8bit, tmp_proxF2_denoise );
	tmp_proxF2_denoise.convertTo(proxF[2], CV_64F);

	for (int k = 0; k < 3; k++) {
		imwrite ("Flex_output/porxF_" + int2str(k) + ".png", proxF[k]);
	}

	// proximal-gamma-F*(v) = v - gamma * proximal-inv_gamma-F(v/gamma)
	for (int k = 0; k < 3; k++) {
		y[k] = v[k] - (gamma * proxF[k]);
	}
	for (int k = 0; k < 3; k++) {
		imwrite ("Flex_output/penalty_" + int2str(k) + ".png", y[k]);
	}

}


void data_fidelity (Mat& x_k1, Mat& x_k, vector<Mat>& y, double tau) {
	Mat grad_x, grad_y;
	Sobel( y[0], grad_x, CV_64F, 1, 0, 3, 1, 0, BORDER_DEFAULT );
	Sobel( y[1], grad_y, CV_64F, 0, 1, 3, 1, 0, BORDER_DEFAULT );

	Mat v = Mat::zeros(x_k.rows, x_k.cols, CV_64F);
	v = x_k - (tau * (- grad_x - grad_y + y[2]));	// whether this grad_x,grad_y should be + or -, need to be try

	imwrite("Flex_output/data.png", v);
}

void form_tauATAplusI (double tau, vector<EigenSpMat>& ST, vector<Mat>& conf, vector<EigenSpMat>& S, EigenSpMat& out) {
	// this is calclulating ST * conf * S

	out = EigenSpMat(S[0].cols(), S[0].cols());
	vector<T> tripletList;
	tripletList.reserve(6553600);

	int LR_cols = conf[0].cols, LR_rows = conf[0].rows;
	int LR_idx[2], HR_idx[2];
	double tmp_val, tmp_conf;

	// multiply confidence on S
	for (int k = 0; k < ST.size(); k++) {

		for (int tk=0; tk < ST[k].outerSize(); ++tk)
			for (EigenSpMat::InnerIterator it(ST[k],tk); it; ++it)
			{
				LR_idx[0] = it.col() / LR_cols;
				LR_idx[1] = it.col() % LR_cols;

				tmp_conf = conf[k].at<double>(LR_idx[0], LR_idx[1]);

				if (tmp_conf > 100*EXsmall) {
					for (int c = 0; c < (S[k].cols()); c++) {
						tmp_val = S[k].coeffRef (it.col(), c);
						if (tmp_val > 100*EXsmall) {
							out.coeffRef(it.row(), c) += tau * it.value() * tmp_conf * tmp_val;
						}
					}
				}

			}
	}

}

void form_tauATz (double tau, vector<EigenSpMat>& ST, vector<Mat>& conf, vector<Mat>& LRimgs, Mat& out, int HR_rows, int HR_cols) {

	out = Mat::zeros(HR_rows, HR_cols, CV_64F);

	int LR_cols = LRimgs[0].cols, LR_rows = LRimgs[0].rows;
	int cur_row, colIdx2Pos;
	int LR_idx[2], HR_idx[2];

	for (int k = 0; k < ST.size(); k++) {

		for (int tk=0; tk < ST[k].outerSize(); ++tk)
			for (EigenSpMat::InnerIterator it(ST[k],tk); it; ++it)
			{
				LR_idx[0] = it.col() / LR_cols;
				LR_idx[1] = it.col() % LR_cols;
				
				HR_idx[0] = it.row() / HR_cols;
				HR_idx[1] = it.row() % HR_cols;

				out.at<double>(HR_idx[0], HR_idx[1]) += it.value() * conf[k].at<double>(LR_idx[0], LR_idx[1]) * LRimgs[k].at<double>(LR_idx[0], LR_idx[1]);
			}
	}
}