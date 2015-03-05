#include "Methods.h"

// BP
void formInfluenceRelation (vector<Mat>& imgs,
							vector<Mat>& flows,
							vector < vector < vector <LR_Pixel> > >& LR_pixels,
							vector < vector <HR_Pixel> >&  HR_pixels,
							double scale,
							Mat& super_PSF,
							Mat& super_BPk,
							double interp_scale) {
	int i, j, k;
	int x, y;
	
	int LR_rows = imgs[0].rows;
	int LR_cols = imgs[0].cols;
	int HR_rows = LR_rows * scale;
	int HR_cols = LR_cols * scale;
	
	int PSF_radius_x = super_PSF.cols / interp_scale / 2;
	int PSF_radius_y = super_PSF.rows / interp_scale / 2;
	int BPk_radius_x = super_BPk.cols / interp_scale / 2;
	int BPk_radius_y = super_BPk.rows / interp_scale / 2;

	double pos_x, pos_y, bucket_center_x, bucket_center_y, dist_x, dist_y, dx, dy, offset_x, offset_y;;
	int bucket_idx_i, bucket_idx_j, super_offset_x, super_offset_y;

	

	// star record
	// for each image
	for (k = 0; k < imgs.size(); k++) {
		// for each pixel
		for (i = 0; i < LR_rows; i++) {
			for (j = 0; j < LR_cols; j++) {
				Vec2f& tmp_flow = flows[k].at<Vec2f>(i,j);
				pos_x = (j + tmp_flow[0] + 0.5) * scale;
				pos_y = (i + tmp_flow[1] + 0.5 ) * scale;

				LR_pixels[k][i][j].val = (double)imgs[k].at<uchar>(i,j);
				LR_pixels[k][i][j].pos_x = pos_x;
				LR_pixels[k][i][j].pos_y = pos_y;

				// add to those buckets within radius
				// for each possible bucket
				for (y = -PSF_radius_y-1; y < PSF_radius_y + 3; y++) {
					for (x = -PSF_radius_x-1; x < PSF_radius_y + 3; x++) {
						bucket_idx_i = pos_y + y;
						bucket_idx_j = pos_x + x;
						bucket_center_x = bucket_idx_j + 0.5;
						bucket_center_y = bucket_idx_i + 0.5;
						// check if bucket exist
						if (bucket_center_x < 0 || bucket_center_y < 0 || bucket_center_y >= HR_rows || bucket_center_x >= HR_cols)
							continue;
						// check if within PSF_radius
						dx = pos_x - bucket_center_x;
						dy = pos_y - bucket_center_y;
						dist_x = abs(dx);
						dist_y = abs(dy);
						if (dist_x-0.5 > PSF_radius_x || dist_y-0.5 > PSF_radius_y)
							continue;

						// create a influence relation
						Influenced_Pixel tmp_pix;
						tmp_pix.pixel = &(LR_pixels[k][i][j]);
						//----- hbp
						offset_x = (dx) + BPk_radius_x + 0.5;
						offset_y = (dy) + BPk_radius_y + 0.5;
						// if offset is just on the edge of PSF
						if (offset_x == BPk_radius_x * 2 + 1) offset_x -= EXsmall;
						if (offset_y == BPk_radius_y * 2 + 1) offset_y -= EXsmall;
						super_offset_x = offset_x * interp_scale;
						super_offset_y = offset_y * interp_scale;
						tmp_pix.hBP = super_BPk.at<double>(super_offset_x, super_offset_y);
						// add to bucket
						HR_pixels[ bucket_idx_i ][ bucket_idx_j ].influenced_pixels.push_back( tmp_pix );
						HR_pixels[ bucket_idx_i ][ bucket_idx_j ].hBP_sum += tmp_pix.hBP;

						// create a perception relation
						Perception_Pixel tmp_pix2;
						tmp_pix2.pixel = &(HR_pixels[ bucket_idx_i ][ bucket_idx_j ]);
						// ----- hpsf
						offset_x = (dx) + PSF_radius_x + 0.5;
						offset_y = (dy) + PSF_radius_y + 0.5;
						if (offset_x == PSF_radius_x * 2 + 1) offset_x -= EXsmall;
						if (offset_y == PSF_radius_y * 2 + 1) offset_y -= EXsmall;
						super_offset_x = offset_x * interp_scale;
						super_offset_y = offset_y * interp_scale;
						tmp_pix2.hPSF = super_PSF.at<double>(super_offset_x, super_offset_y);

						LR_pixels[k][i][j].perception_pixels.push_back(tmp_pix2);
					}
				}
			}
		}
	}

}

void preInterpolation ( Mat& PSF, Mat& super_PSF, double PSF_scale )
{
	// make sure the skirt of PSF is ZERO!!
	Mat tmp_PSF = Mat::zeros(PSF.rows+2, PSF.cols+2, CV_64F);
	for (int i = 0; i < PSF.rows; i ++) for (int j = 0; j < PSF.cols; j++)
		tmp_PSF.at<double>(i+1,j+1) = PSF.at<double>(i,j);

	resize(tmp_PSF, super_PSF, Size(0,0), PSF_scale, PSF_scale, INTER_CUBIC);
}

void HR_to_LR ( Mat& HRimg, Mat& LRimg, double scale, Mat& PSF, bool is_super_PSF, double PSF_scale )
{

	int PSF_radius_x, PSF_radius_y;

	Mat super_PSF;
	LRimg = Mat::zeros( HRimg.rows / scale, HRimg.cols / scale, CV_64F);

	// directly use imresize to have a approximate pre-interpolation of PSF
	if (is_super_PSF) {
		super_PSF = PSF;
	}
	else {
		PSF_scale = 1000;
		preInterpolation (PSF, super_PSF, PSF_scale);
	}

	PSF_radius_x = super_PSF.cols / PSF_scale / 2;
	PSF_radius_y = super_PSF.rows / PSF_scale / 2;

	double pos_x, pos_y;
	int cur_pix_i, cur_pix_j;
	int locate_pixel_i, locate_pixel_j;
	double cur_pix_pos_x, cur_pix_pos_y; 
	double PSF_left, PSF_top;
	double offset_x, offset_y, dist_x, dist_y;
	int super_offset_x, super_offset_y;

	for (int i = 0; i < LRimg.rows; i++) {
		for (int j = 0; j < LRimg.cols; j++) {
			pos_y = (i + 0.5)*scale;
			pos_x = (j + 0.5)*scale;

			locate_pixel_i = pos_y;
			locate_pixel_j = pos_x;

			// for all posiible neighbor pixel
			for (int y = -PSF_radius_y-1; y < PSF_radius_y+3; y++ ) {
				for (int x = -PSF_radius_x-1; x < PSF_radius_x+3; x++) {
					cur_pix_pos_x = locate_pixel_j + 0.5 + x;
					cur_pix_pos_y = locate_pixel_i + 0.5 + y;
					cur_pix_i = cur_pix_pos_y;
					cur_pix_j = cur_pix_pos_x;

					// if pixel is out of bound, it means it can't be sampled
					if ( cur_pix_pos_x < 0 || cur_pix_pos_x >= HRimg.cols || cur_pix_pos_y < 0 || cur_pix_pos_y >= HRimg.rows )
						continue;

					dist_x = abs(pos_x - cur_pix_pos_x);
					dist_y = abs(pos_y - cur_pix_pos_y);
					// if dist is bigger than PSF_radius
					if (dist_x-0.5 > PSF_radius_x || dist_y-0.5 > PSF_radius_y)
						continue;

					PSF_left = cur_pix_pos_x - PSF_radius_x - 0.5;
					PSF_top = cur_pix_pos_y - PSF_radius_y - 0.5;

					offset_x = pos_x - PSF_left ;
					offset_y = pos_y - PSF_top ;

					// if offset is just on the edge of PSF
					if (offset_x == PSF_radius_x * 2 + 1) offset_x -= EXsmall;
					if (offset_y == PSF_radius_y * 2 + 1) offset_y -= EXsmall;

					super_offset_x = offset_x * PSF_scale;
					super_offset_y = offset_y * PSF_scale;

					LRimg.at<double>	(i,j) +=super_PSF.at<double>( super_offset_y, super_offset_x ) * (HRimg.at<double>(cur_pix_i, cur_pix_j)  );

				}
			}

		}
	}
}

void HR_to_LR_percetion ( Mat& HRimg, vector < vector < vector <LR_Pixel> > >& LR_pixels, double scale, Mat& PSF, bool is_super_PSF, double PSF_scale )
{

	int PSF_radius_x, PSF_radius_y;

	Mat super_PSF;

	// directly use imresize to have a approximate pre-interpolation of PSF
	if (is_super_PSF) {
		super_PSF = PSF;
	}
	else {
		PSF_scale = 1000;
		preInterpolation (PSF, super_PSF, PSF_scale);
	}

	PSF_radius_x = super_PSF.cols / PSF_scale / 2;
	PSF_radius_y = super_PSF.rows / PSF_scale / 2;

	double pos_x, pos_y;
	int cur_pix_i, cur_pix_j;
	int locate_pixel_i, locate_pixel_j;
	double cur_pix_pos_x, cur_pix_pos_y; 
	double PSF_left, PSF_top;
	double offset_x, offset_y, dist_x, dist_y;
	int super_offset_x, super_offset_y;

	for (int k = 0; k < LR_pixels.size(); k++) {
		for (int i = 0; i < LR_pixels[k].size(); i++) {
			for (int j = 0; j < LR_pixels[k][i].size(); j++) {
				LR_pixels[k][i][j].perception = 0;

				pos_y = LR_pixels[k][i][j].pos_y;
				pos_x = LR_pixels[k][i][j].pos_x;

				locate_pixel_i = pos_y;
				locate_pixel_j = pos_x;

				// for all posiible neighbor pixel
				for (int y = -PSF_radius_y-1; y < PSF_radius_y+3; y++ ) {
					for (int x = -PSF_radius_x-1; x < PSF_radius_x+3; x++) {
						cur_pix_pos_x = locate_pixel_j + 0.5 + x;
						cur_pix_pos_y = locate_pixel_i + 0.5 + y;
						cur_pix_i = cur_pix_pos_y;
						cur_pix_j = cur_pix_pos_x;

						// if pixel is out of bound, it means it can't be sampled
						if ( cur_pix_pos_x < 0 || cur_pix_pos_x >= HRimg.cols || cur_pix_pos_y < 0 || cur_pix_pos_y >= HRimg.rows )
							continue;

						dist_x = abs(pos_x - cur_pix_pos_x);
						dist_y = abs(pos_y - cur_pix_pos_y);
						// if dist is bigger than PSF_radius
						if (dist_x-0.5 > PSF_radius_x || dist_y-0.5 > PSF_radius_y)
							continue;

						PSF_left = cur_pix_pos_x - PSF_radius_x - 0.5;
						PSF_top = cur_pix_pos_y - PSF_radius_y - 0.5;

						offset_x = pos_x - PSF_left ;
						offset_y = pos_y - PSF_top ;

						// if offset is just on the edge of PSF
						if (offset_x == PSF_radius_x * 2 + 1) offset_x -= EXsmall;
						if (offset_y == PSF_radius_y * 2 + 1) offset_y -= EXsmall;

						super_offset_x = offset_x * PSF_scale;
						super_offset_y = offset_y * PSF_scale;

						LR_pixels[k][i][j].perception += super_PSF.at<double>( super_offset_y, super_offset_x ) * (HRimg.at<double>(cur_pix_i, cur_pix_j)  );
					}
				}

			}
		}
	}
}
/**/
// OptFlow
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

void showConfidence (Mat& flow_forward, Mat& flow_backward, Mat& confidence)
{
	// use CV_32FC2

	confidence = Mat::zeros(flow_forward.rows, flow_forward.cols, CV_64F);

	int i, j;

	Mat forw_pos = Mat::zeros(confidence.rows, confidence.cols, CV_32FC2);

	for (i = 0; i < confidence.rows; i++) for (j = 0; j < confidence.cols; j++) {
		// forward
		Vec2f& tmp_pos = forw_pos.at<Vec2f>(i, j);
		Vec2f& tmp_flow= flow_forward.at<Vec2f>(i, j);

		tmp_pos[0] = i + tmp_flow[1];
		tmp_pos[1] = j + tmp_flow[0];
	}

	float max_confdence = 0;

	float portion_i, portion_j;
	int i_to, j_to;
	int n, m;
	float sum;

	for (i = 0; i < confidence.rows; i++) for (j = 0; j < confidence.cols; j++) {
		Vec2f& cur_flow = flow_forward.at<Vec2f>(i, j);
		Vec2f& tmp_forwpos = forw_pos.at<Vec2f>(i, j);

		i_to = tmp_forwpos[0];
		j_to = tmp_forwpos[1];

		if (i_to < 0 || j_to < 0 || i_to+1 >= confidence.rows || j_to+1 >= confidence.cols) {
			if (i_to < 0 || j_to < 0 || i_to >= confidence.rows || j_to >= confidence.cols) continue;

			// for boundary case
			if (i_to+1 < confidence.rows) {
				// j_to == confidence.cols -1
				portion_i = 1 - (tmp_forwpos[0] - i_to);
				Vec2f& ij = flow_backward.at<Vec2f> (i_to, j_to);
				Vec2f& i1j = flow_backward.at<Vec2f> (i_to+1, j_to);
				confidence.at<double>(i, j) = 
					portion_i * calcConfidence(cur_flow, ij) + 
					(1-portion_i) * calcConfidence(cur_flow, i1j);
			}
			else if (j_to+1 < confidence.cols) {
				// i_to == confidence.rows - 1

				portion_j = 1 - (tmp_forwpos[1] - j_to);
				Vec2f& ij = flow_backward.at<Vec2f> (i_to, j_to);
				Vec2f& ij1 = flow_backward.at<Vec2f> (i_to, j_to+1);
				confidence.at<double>(i, j) = 
					portion_j * calcConfidence(cur_flow, ij) + 
					(1-portion_j) * calcConfidence(cur_flow, ij1);
			}
			else {
				// i_to == confidence.rows - 1 && j_to == confidence.cols -1
				Vec2f& ij = flow_backward.at<Vec2f> (i_to, j_to);

				confidence.at<double>(i, j) = 
					calcConfidence(cur_flow, ij);
			}


			continue;
		}

		portion_i = 1 - (tmp_forwpos[0] - i_to);
		portion_j = 1 - (tmp_forwpos[1] - j_to);

		Vec2f& ij = flow_backward.at<Vec2f> (i_to, j_to);
		Vec2f& i1j = flow_backward.at<Vec2f> (i_to+1, j_to);
		Vec2f& ij1 = flow_backward.at<Vec2f> (i_to, j_to+1);
		Vec2f& i1j1 = flow_backward.at<Vec2f> (i_to+1, j_to+1);

		confidence.at<double>(i, j) = 
			portion_i * portion_j * calcConfidence(cur_flow, ij) + 
			(1-portion_i) * portion_j * calcConfidence(cur_flow, i1j) + 
			portion_i * (1-portion_j) * calcConfidence(cur_flow, ij1) +
			(1-portion_i) * (1-portion_j) * calcConfidence(cur_flow, i1j1); 

		// it's no longer useful, I guess max_confidence = 1
		if (confidence.at<double>(i, j) > max_confdence) max_confdence = confidence.at<double>(i, j);
	}

	confidence = confidence /** 254*/ + EXsmall;
	//confidence = EXsmall;
}

double ExpNegSQR (float x, float y) {
	float sigma = 1;
	return exp(-(SQR(x) + SQR(y)) / sigma);
}

double calcConfidence (Vec2f& f, Vec2f& b)
{
	float diff_x, diff_y;
	diff_x = f[0] + b[0];
	diff_y = f[1] + b[1];

	return ExpNegSQR(diff_x, diff_y);
}
// FlexISP
void resampleByMatrix (Mat& X,
					   vector <EigenSpMat>& S, 
					   vector <Mat>& SX,
					   int LR_Rows,
					   int LR_Cols) {
	int LR_ImgCount = S.size(),
		HR_Rows = X.rows,
		HR_Cols = X.rows;
	SX.resize(LR_ImgCount);

	int LR_idx[2], HR_idx[2];

	for (int k = 0; k < LR_ImgCount; k++) {
		SX[k] = Mat::zeros(LR_Rows, LR_Cols, CV_64F);

		for (int tk=0; tk < S[k].outerSize(); ++tk)
			for (EigenSpMat::InnerIterator it(S[k],tk); it; ++it)
			{
				LR_idx[0] = it.row() / LR_Cols;
				LR_idx[1] = it.row() % LR_Cols;
				
				HR_idx[0] = it.col() / HR_Cols;
				HR_idx[1] = it.col() % HR_Cols;

				SX[k].at<double>(LR_idx[0], LR_idx[1]) += it.value() * X.at<double>(HR_idx[0], HR_idx[1]);
			}
	}
}

void formSparseI (EigenSpMat& out, int rows, int cols) {
	out = EigenSpMat(rows, cols);

	int I_length = MIN(rows, cols);

	vector<T> tripletList;
	tripletList.reserve(I_length);

	for (int i = 0; i < I_length; i++) {
		tripletList.push_back(T(i, i, 1));
	}

	out.setFromTriplets(tripletList.begin(), tripletList.end());
}