#include "BP_Reconstruction.h"

BP_Constructor::BP_Constructor( Mat& HRimg, double scale, vector<Mat>& imgs, vector<Mat>& flows, Mat& PSF, Mat& BPk, TermCriteria& BPstop )
{
	vector<Mat> confs;
	confs.resize(imgs.size());
	confs[0] = Mat::ones(imgs[0].size(), CV_64F);
	for(int i = 0; i < confs.size(); i++) {
		confs[i] = confs[0];
	}

	constructor(HRimg, scale, imgs, flows,  PSF,  BPk, BPstop, confs);
	
	solve();

	output(HRimg);
}

BP_Constructor::BP_Constructor( Mat& HRimg, double scale, vector<Mat>& imgs, vector<Mat>& flows, Mat& PSF, Mat& BPk, TermCriteria& BPstop,  vector<Mat>& confs )
{
	constructor(HRimg, scale, imgs, flows,  PSF,  BPk, BPstop, confs);
	
	solve();

	output(HRimg);
}

void BP_Constructor::constructor( Mat& HRimg, double scale, vector<Mat>& imgs, vector<Mat>& flows, Mat& PSF, Mat& BPk, TermCriteria& BPstop, vector<Mat>& confs )
{

	// all LR images should have same Size()

	// parameter
	BP_c = 0.5;
	interp_scale = 500;

	int i, j, k;

	LR_rows = imgs[0].rows;
	LR_cols = imgs[0].cols;
	HR_rows = LR_rows * scale;
	HR_cols = LR_cols * scale;
	LR_imgCount = imgs.size();

	// pre-interpolation
	Mat super_PSF;
	Mat super_BPk;	
	preInterpolation ( PSF, super_PSF, interp_scale);
	preInterpolation ( BPk, super_BPk, interp_scale);

	//imwrite("output/PSF.bmp", super_PSF*255);

	//----- for every pixel x of HR image, record influenced pixel y
	// initialize bucket
	HR_pixels = new HR_Pixel_Array(HR_rows, HR_cols);
	// initialize influenced pixels (for each pixel in each LR img)
	LR_pixels = new LR_Pixel_Array(LR_imgCount, LR_rows, LR_cols);
	//
	relations = new InfluenceRelation (imgs,
							flows,
							LR_pixels,
							HR_pixels,
							scale,
							super_PSF,
							super_BPk,
							interp_scale,
							confs);
	//
	setHRinitial();

	//
	BPtermination = BPstop;
	BPtermination.type = BPstop.type;
	BPtermination.maxCount = BPstop.maxCount;
	BPtermination.epsilon = BPstop.epsilon;
}

void BP_Constructor::setHRinitial()
{
	int i, j, k;

	double tmp_sum;
	for (i = 0; i < HR_rows; i++) {
		for (j = 0; j < HR_cols; j++) {
			tmp_sum = 0;

			for (k = 0; k < HR_pixels->access(i, j).influence_link_cnt; k++) {
				tmp_sum += relations->influence_links[HR_pixels->access(i, j).influence_link_start + k].pixel->val;
			}
			if (HR_pixels->access(i, j).influence_link_cnt == 0) {
				HR_pixels->access(i, j).val = tmp_sum;
				//HRimg.at<double>(i,j) = tmp_sum;
			}
			else {
				HR_pixels->access(i, j).val = tmp_sum / HR_pixels->access(i, j).influence_link_cnt;
				//HRimg.at<double>(i,j) = tmp_sum / HR_pixels->access(i, j).influence_link_cnt;
			}
			//for (k = 0; k < HR_pixels->access(i, j).influenced_pixels.size(); k++) {
			//	tmp_sum += HR_pixels->access(i, j).influenced_pixels[k].pixel->val;
			//}
			//if (HR_pixels->access(i, j).influenced_pixels.size() == 0) 
			//	HRimg.at<double>(i,j) = tmp_sum;
			//else
			//	HRimg.at<double>(i,j) = tmp_sum / HR_pixels->access(i, j).influenced_pixels.size();
		}
	}
	/**/
}

void BP_Constructor::solve()
{
	int i,j,k;

	double sum_diff, sum_hBP, sum_confidence;
	double cur_hBP, cur_confidence, diff;

	Mat LR_per;
	// start Back Projection
	Mat HRimg = Mat::zeros(HR_rows, HR_cols, CV_64F), Lap = Mat::zeros(HR_rows, HR_cols, CV_64F);
	vector<double> tmp_vec;
	tmp_vec.resize(4);

	int iter = 1;
	double epsi = 0;
	bool stop = false;
	while (!stop) {
		epsi = 0;

		HR_to_LR_percetion(*HR_pixels, *LR_pixels, *relations);
		
		
		// try also update laplace term
		for (i = 0; i < HR_rows; i++) for (j = 0; j < HR_cols; j++)
		{
			HRimg.at<double>(i, j) = HR_pixels->access(i, j).val;
		}
		//Laplacian(HRimg, Lap, CV_64F, 1, 1, 0, BORDER_REPLICATE );

		// see LR perception
		//
		//LR_per = Mat::zeros(LR_rows, LR_cols, CV_64F);
		//for (k = 0; k < imgs.size(); k++) {
		//	for (i = 0; i < LR_rows; i++) for (j = 0; j < LR_cols; j++) {
		//		LR_per.at<double>(i,j) = LR_pixels[k][i][j].percetion;
		//	}
		//	String LR_p_name = "output/LR_P_" + int2str(k) + "_" + int2str(iter) + ".bmp";
		//	imwrite(LR_p_name, LR_per);
		//}
		//


		// for each HR x, calculate f(x)'n+1'
		for (i = 1; i < HR_rows-1; i++) for (j = 1; j < HR_cols-1; j++) {
			sum_diff = 0;
			sum_hBP = HR_pixels->access(i, j).hBP_sum;
			sum_confidence = 0;
			// for all influenced pixel
			for (k = 0; k < HR_pixels->access(i, j).influence_link_cnt; k++) {
				Influenced_Pixel& cur_influenced_pix = relations->influence_links[HR_pixels->access(i, j).influence_link_start + k];
				cur_hBP = cur_influenced_pix.hBP;
				cur_confidence = cur_influenced_pix.pixel -> confidence;
				// sum up hBP
				//sum_hBP += cur_hBP;
				// sum up confidence weight
				sum_confidence += cur_confidence;

				// sum up diff
				diff = cur_influenced_pix.pixel->val - cur_influenced_pix.pixel->perception;				
				sum_diff += diff * ( SQR(cur_hBP) ) * cur_confidence/**/;	
				//sum_diff += diff *  cur_hBP  * cur_confidence;	
			}
			// deal with constant & weight normalization
			sum_diff /= (BP_c * sum_hBP /** sum_confidence/**/);

			// try also update laplace
			//sum_diff += 0.05 * (Lap.at<double>(i, j));

			// try heuristic L1
			
			tmp_vec[0] = HRimg.at<double>(i-1, j);
			tmp_vec[1] = HRimg.at<double>(i+1, j);
			tmp_vec[2] = HRimg.at<double>(i, j-1);
			tmp_vec[3] = HRimg.at<double>(i, j+1);

			sort(tmp_vec.begin(), tmp_vec.end(), compareDouble);
			double new_x = HR_pixels->access(i, j).val  + sum_diff;
			double cur_x = HR_pixels->access(i, j).val;
			double L1_reg = 0;
			/*
			if (cur_x < tmp_vec[1]) {
				L1_reg = tmp_vec[1] - cur_x;
			}
			else if (cur_x > tmp_vec[2]) {
				L1_reg = tmp_vec[2] - cur_x;
			}
			//*/
			/*
			if (cur_x < tmp_vec[1]) {
				L1_reg = 0.75 * (tmp_vec[1] - cur_x) + 0.25 * (tmp_vec[2] - cur_x);
			}
            if (cur_x > tmp_vec[2]) {
				L1_reg = 0.75 * (tmp_vec[2] - cur_x) + 0.25 * (tmp_vec[3] - cur_x);
			}
			//*/
			if (cur_x < tmp_vec[0]) {
				L1_reg = tmp_vec[1] - cur_x;
			}
			else if (cur_x < tmp_vec[1]) {
				L1_reg = tmp_vec[2] - cur_x;
			}
			if (cur_x > tmp_vec[3]) {
				L1_reg = tmp_vec[2] - cur_x;
			}
			else if (cur_x > tmp_vec[2]) {
				L1_reg = tmp_vec[1] - cur_x;
			}

			sum_diff += 0.05 * L1_reg;
			//*/

			// update HR
			HR_pixels->access(i, j).val += sum_diff;
			//HRimg.at<double>(i,j) += sum_diff;

			// update epsi = max (sum_diff)
			if (abs(sum_diff) > epsi) epsi = abs(sum_diff);
		}

		//imwrite("output/HR_BP_" + int2str(iter) + ".bmp", HRimg);

		// check termination criterion
		if (BPtermination.type == TermCriteria::COUNT) {
			if (iter >= BPtermination.maxCount) stop = true;
		}
		else if (BPtermination.type == TermCriteria::EPS) {
			if (epsi < BPtermination.epsilon) stop = true;	
		}
		else if (BPtermination.type == TermCriteria::EPS + TermCriteria::COUNT) {
			if (iter >= BPtermination.maxCount || epsi < BPtermination.epsilon) stop = true;
		}

		iter++;

		cout << "BP iteration: " << iter << endl;
		cout << "epsilon: " << epsi << endl;
	}

	cout << "BP iteration: " << iter << endl;
	cout << "epsilon: " << epsi << endl;
	/**/

}

void BP_Constructor::output(Mat& HRimg)
{
	HRimg = Mat::zeros(HR_rows, HR_cols, CV_64F);

	for (int i = 0; i < HRimg.rows; i++) for (int j = 0; j < HRimg.cols; j++) {
		HRimg.at<double>(i, j) = HR_pixels->access(i, j).val;
	}
}

BP_Constructor::~BP_Constructor()
{
	delete HR_pixels;
	delete LR_pixels;
	delete relations;
}

bool compareDouble (double a, double b)
{
	return (a < b);
}