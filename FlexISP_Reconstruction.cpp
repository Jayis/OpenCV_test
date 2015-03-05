#include "FlexISP_Reconstruction.h"

void FlexISPmain (vector<Mat>& imgsC1, vector<Mat>& flows, vector<Mat>& confs, Mat& PSF, Mat& BPk, double scale, Mat& output) {
	double gamma, tau, theta;

	int LR_rows = imgsC1[0].rows;
	int LR_cols = imgsC1[0].cols;
	int HR_rows = LR_rows * scale;
	int HR_cols = LR_cols * scale;

	Mat super_PSF;
	Mat super_BPk;	
	preInterpolation ( PSF, super_PSF, 1000);
	preInterpolation ( BPk, super_BPk, 1000);

	// initialize HR pixel (for relation use)
	vector < vector < HR_Pixel> >  HR_pixels;
	HR_pixels.resize(HR_rows);
	for (int i = 0; i < HR_pixels.size(); i++) {
		HR_pixels[i].resize(HR_cols);
	}
	for (int i = 0; i < HR_pixels.size(); i++) for (int j = 0; j < HR_pixels[0].size(); j++) {
		HR_pixels[i][j].i = i;
		HR_pixels[i][j].j = j;
	}
	// initialize influenced pixels (for each pixel in each LR img)
	vector < vector < vector <LR_Pixel> > > LR_pixels;
	LR_pixels.resize(imgsC1.size());
	for (int k = 0; k < imgsC1.size(); k++) {
		LR_pixels[k].resize(LR_rows);
		for (int i = 0; i < LR_rows; i++) {
			LR_pixels[k][i].resize(LR_cols);
		}
	}
	for (int k = 0; k < imgsC1.size(); k++) for (int i = 0; i < LR_rows; i++) for (int j = 0; j < LR_cols; j++) {
		LR_pixels[k][i][j].i = i;
		LR_pixels[k][i][j].j = j;
		LR_pixels[k][i][j].k = k;
	}
	formInfluenceRelation (imgsC1, flows, LR_pixels, HR_pixels, scale, super_PSF, super_BPk,1000);

	// use relation to form resample matrix
	// include scale, blur, 
	vector<EigenSpMat> S, ST;
	formResampleMatrix (LR_pixels, HR_pixels, S, ST);

	// form matrix for linear system in data fidelity
	Mat tauATz;
	EigenSpMat tauATAplusI;
	form_tauATz (tau, ST, confs, imgsC1, tauATz, HR_rows, HR_cols);
	form_tauATAplusI (tau, ST, confs, S, tauATAplusI);
	Mat x_0 = Mat::zeros(HR_rows, HR_cols, CV_64F);
	Mat tmp_HR;
	resize(imgsC1[0], tmp_HR, Size(HR_rows, HR_cols), 0, 0, INTER_CUBIC);
	tmp_HR.convertTo(x_0, CV_64F);
	imwrite("output/HR_cubic.png" ,x_0);

	// start
	FirstOrderPrimalDual (gamma, tau, theta, x_0, tauATz, tauATAplusI, output);

}

void FirstOrderPrimalDual (double gamma, double tau, double theta, Mat& x_0, Mat& tauATz, EigenSpMat& tauATAplusI, Mat& result) {
	int HR_cols = x_0.cols, HR_rows = x_0.rows;
	
	

	// initialize tmp_x[2]
	bool turn = false; // false = 0, true = 1
	Mat tmp_x[2];
	tmp_x[0] = Mat::zeros(HR_rows, HR_cols, CV_64F);
	tmp_x[1] = Mat::zeros(HR_rows, HR_cols, CV_64F);

	// initialize x_bar_0
	Mat x_bar = Mat::zeros(HR_rows, HR_cols, CV_64F);
	x_0.copyTo(x_bar);

	// initialize y
	vector<Mat> y;
	y.resize(3);
	for (int i = 0; i < 3; i++) {
		y[i] = Mat::zeros(HR_rows, HR_cols, CV_64F);
	}

	// pre- calculate
	ConjugateGradient<EigenSpMat> cg;
	cg.solve(tauATAplusI);

	// start
	int cur, next;
	double max_diff = 3e8, cur_diff;

	while ( max_diff > 1 ) {
		max_diff = 0;

		if (turn) {
			cur = 1;
			next = 0;
		}
		else {
			cur = 0; 
			next = 1;
		}

		Mat& x_k = tmp_x[cur];
		Mat& x_k1 = tmp_x[next];

		penalty (y, x_bar, gamma);
		data_fidelity (x_k1, x_k, y, tau, tauATz, cg);
		extrapolation (x_bar, x_k1, x_k, theta);

		// 
		for (int i = 0; i < x_k.rows; i++) for (int j = 0; j < x_k.cols; j++) {
			cur_diff = abs(x_k.at<double>(i,j) - x_k1.at<double>(i,j));
			if (cur_diff > max_diff) max_diff = cur_diff;
		}
	}

	result = Mat::zeros(HR_rows, HR_cols, CV_64F);
	tmp_x[next].copyTo(result);
}

void penalty (vector<Mat>& y, Mat& x_bar_k, double gamma) {
	// 
	double inv_gamma = 1.0 / gamma;

	//---- v = y + gamma * K * x_bar
	vector<Mat> v;
	v.resize(3);
	for (int k = 0; k < 3; k++) {
		v[k] = Mat::zeros(x_bar_k.rows, x_bar_k.cols, CV_64F);
	}
	
	Mat grad_x, grad_y;
	Sobel( x_bar_k, grad_x, CV_64F, 1, 0, 3, 1, 0, BORDER_DEFAULT );
	Sobel( x_bar_k, grad_y, CV_64F, 0, 1, 3, 1, 0, BORDER_DEFAULT );

	v[0] = y[0] + (gamma * grad_x);
	v[1] = y[1] + (gamma * grad_y);
	v[2] = y[2] + (gamma * x_bar_k);

	for (int k = 0; k < 3; k++) {
		imwrite ("Flex_output/v_" + int2str(k) + ".png", v[k]);
	}

	// calculate proximal-inv_gamma-F(v/gamma)
	// first proximal operator
	// TV 1, image gradient sparsity prior
	vector<Mat> proxF;
	proxF.resize(3);
	for (int k = 0; k < 3; k++) {
		proxF[k] = Mat::zeros(x_bar_k.rows, x_bar_k.cols, CV_64F);
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


void data_fidelity (Mat& x_k1, Mat& x_k, vector<Mat>& y, double tau, Mat& tauATz, ConjugateGradient<EigenSpMat>& cg) {
	Mat grad_x, grad_y;
	Sobel( y[0], grad_x, CV_64F, 1, 0, 3, 1, 0, BORDER_DEFAULT );
	Sobel( y[1], grad_y, CV_64F, 0, 1, 3, 1, 0, BORDER_DEFAULT );

	Mat v = Mat::zeros(x_k.rows, x_k.cols, CV_64F);
	v = x_k - (tau * (- grad_x - grad_y + y[2]));	// whether this grad_x,grad_y should be + or -, need to be try

	imwrite("Flex_output/data.png", v);

	Mat b_Mat = Mat::zeros(x_k.rows, x_k.cols, CV_64F);
	b_Mat = tauATz + v;

	// solve linear system
	// form b
	int HR_vecLength = x_k.rows * x_k.cols;
	int cur_idx = 0;
	VectorXd b(HR_vecLength), x(HR_vecLength);
	for (int i = 0; i < x_k.rows; i++) for (int j = 0; j < x_k.cols; j++) {
		b(cur_idx) = b_Mat.at<double> (i, j);
		cur_idx++;
	}
	// A = tauATAplusI
	// cg.compute(A) should be done outside this function
	// so let's solve	
	x = cg.solve(b);

	// turn x to Mat x_k1
	x_k1 = Mat::zeros(x_k.rows, x_k.cols, CV_64F);
	cur_idx = 0;
	for (int i = 0; i < x_k1.rows; i++) for (int j = 0; j < x_k1.cols; j++) {
		x_k1.at<double> (i, j) = x(cur_idx);
		cur_idx++;
	}
}

void extrapolation (Mat& x_bar_k1, Mat& x_k1, Mat& x_k, double theta) {
	x_bar_k1 = (1 + theta) * x_k1 - x_k;
}

void formResampleMatrix (vector < vector < vector <LR_Pixel> > >& LR_pixels,
							  vector < vector <HR_Pixel> >&  HR_pixels,
							  vector <EigenSpMat>& S,
							  vector <EigenSpMat>& ST) {
 	int LR_ImgCount = LR_pixels.size(),
		LR_Rows = LR_pixels[0].size(),
		LR_Cols = LR_pixels[0][0].size(),
		HR_Rows = HR_pixels.size(),
		HR_Cols = HR_pixels[0].size();
	
	S.resize(LR_ImgCount);
	ST.resize(LR_ImgCount);

	int size[2];
	size[0] = LR_Rows*LR_Cols, size[1] = HR_Rows*HR_Cols;

	int cur_Row, sourcePos2ColIdx, tmp_idx[2];

	vector<T> tripletList, tripletListT;	

	for (int k = 0; k < S.size(); k++) {
		tripletList.reserve(6553600);
		tripletListT.reserve(6553600);
		
		S[k] = EigenSpMat(size[0], size[1]);
		ST[k] = EigenSpMat(size[1], size[0]);

		for (int ii = 0; ii < LR_Rows; ii++) for (int jj = 0; jj < LR_Cols; jj++) {
			cur_Row = ii * LR_Cols + jj;
			tmp_idx[0] = cur_Row;

			for (int p = 0; p < LR_pixels[k][ii][jj].perception_pixels.size(); p++) {
				sourcePos2ColIdx = LR_pixels[k][ii][jj].perception_pixels[p].pixel -> i * HR_Cols +
					LR_pixels[k][ii][jj].perception_pixels[p].pixel -> j;

				tmp_idx[1] = sourcePos2ColIdx;

				tripletList.push_back(T(cur_Row, sourcePos2ColIdx, LR_pixels[k][ii][jj].perception_pixels[p].hPSF));	
				tripletListT.push_back(T(sourcePos2ColIdx, cur_Row, LR_pixels[k][ii][jj].perception_pixels[p].hPSF));	
			}
			
		}
		S[k].setFromTriplets(tripletList.begin(), tripletList.end());
		ST[k].setFromTriplets(tripletListT.begin(), tripletListT.end());

		tripletList.clear();
		tripletListT.clear();
	}
}

void form_tauATAplusI (double tau, vector<EigenSpMat>& ST, vector<Mat>& conf, vector<EigenSpMat>& S, EigenSpMat& out) {
	// this is calclulating ST * conf * S

	out = EigenSpMat(S[0].cols(), S[0].cols());
	EigenSpMat tmp(S[0].cols(), S[0].cols());

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
	EigenSpMat I;
	formSparseI(I, S[0].cols(), S[0].cols());

	out = tmp + I;

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