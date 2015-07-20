#include "Methods.h"

// BP
/*
void formInfluenceRelation (vector<Mat>& imgs,
							vector<Mat>& flows,
							LR_Pixel_Array* LR_pixels,
							HR_Pixel_Array*  HR_pixels,
							double scale,
							Mat& super_PSF,
							Mat& super_BPk,
							double interp_scale) {
	cout << "formInfluenceRelation" << endl;

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
	
	// start record
	// for each image
	for (k = 0; k < imgs.size(); k++) {
		// for each pixel
		for (i = 0; i < LR_rows; i++) {
			for (j = 0; j < LR_cols; j++) {
				Vec2f& tmp_flow = flows[k].at<Vec2f>(i,j);
				pos_x = (j + tmp_flow[0] + 0.5) * scale;
				pos_y = (i + tmp_flow[1] + 0.5 ) * scale;

				LR_pixels->access(k, i, j).val = (double)imgs[k].at<uchar>(i,j);
				LR_pixels->access(k, i, j).pos_x = pos_x;
				LR_pixels->access(k, i, j).pos_y = pos_y;

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
						tmp_pix.pixel = &(LR_pixels->access(k, i, j));
						//----- hbp
						offset_x = (dx) + BPk_radius_x + 0.5;
						offset_y = (dy) + BPk_radius_y + 0.5;
						// if offset is just on the edge of PSF
						if (offset_x == BPk_radius_x * 2 + 1) offset_x -= EX_small;
						if (offset_y == BPk_radius_y * 2 + 1) offset_y -= EX_small;
						super_offset_x = offset_x * interp_scale;
						super_offset_y = offset_y * interp_scale;
						tmp_pix.hBP = super_BPk.at<double>(super_offset_x, super_offset_y);
						// add to bucket
						HR_pixels->access( bucket_idx_i, bucket_idx_j).influenced_pixels.push_back( tmp_pix );
						HR_pixels->access( bucket_idx_i, bucket_idx_j).hBP_sum += tmp_pix.hBP;

						// create a perception relation
						Perception_Pixel tmp_pix2;
						tmp_pix2.pixel = &(HR_pixels->access( bucket_idx_i, bucket_idx_j));
						// ----- hpsf
						offset_x = (dx) + PSF_radius_x + 0.5;
						offset_y = (dy) + PSF_radius_y + 0.5;
						if (offset_x == PSF_radius_x * 2 + 1) offset_x -= EX_small;
						if (offset_y == PSF_radius_y * 2 + 1) offset_y -= EX_small;
						super_offset_x = offset_x * interp_scale;
						super_offset_y = offset_y * interp_scale;
						tmp_pix2.hPSF = super_PSF.at<double>(super_offset_x, super_offset_y);

						LR_pixels->access(k, i, j).perception_pixels.push_back(tmp_pix2);
					}
				}
			}
		}
	}

}
*/

void preInterpolation ( Mat& PSF, Mat& super_PSF, double PSF_scale )
{
	cout << "interpolate with scale: " << PSF_scale << "\n";
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
					if (offset_x == PSF_radius_x * 2 + 1) offset_x -= EX_small;
					if (offset_y == PSF_radius_y * 2 + 1) offset_y -= EX_small;

					super_offset_x = offset_x * PSF_scale;
					super_offset_y = offset_y * PSF_scale;

					LRimg.at<double>(i,j) +=super_PSF.at<double>( super_offset_y, super_offset_x ) * (HRimg.at<double>(cur_pix_i, cur_pix_j)  );

				}
			}

		}
	}
}

void HR_to_LR_percetion ( HR_Pixel_Array& HR_pixels, LR_Pixel_Array& LR_pixels, InfluenceRelation& relations/*, Mat& PSF, bool is_super_PSF, double PSF_scale*/ )
{

	int PSF_radius_x, PSF_radius_y;

	Mat super_PSF;

	/*
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
	*/

	double pos_x, pos_y;
	int cur_pix_i, cur_pix_j;
	int locate_pixel_i, locate_pixel_j;
	double cur_pix_pos_x, cur_pix_pos_y; 
	double PSF_left, PSF_top;
	double offset_x, offset_y, dist_x, dist_y;
	int super_offset_x, super_offset_y;

	for (int k = 0; k < LR_pixels.LR_imgCount; k++) {
		for (int i = 0; i < LR_pixels.LR_rows; i++) {
			for (int j = 0; j < LR_pixels.LR_cols; j++) {
				LR_Pixel& cur_LR_Pixel = LR_pixels.access(k, i, j);

				cur_LR_Pixel.perception = 0;

				/*
				pos_y = LR_pixels[k][i][j].pos_y;
				pos_x = LR_pixels[k][i][j].pos_x;

				locate_pixel_i = pos_y;
				locate_pixel_j = pos_x;
				*/

				for (int t = 0; t < cur_LR_Pixel.perception_link_cnt; t++) {
					Perception_Pixel& cur_perception_pix = relations.perception_links[cur_LR_Pixel.perception_link_start + t];
					cur_LR_Pixel.perception +=
						cur_perception_pix.hPSF *
						//HRimg.at<double>(cur_perception_pix.pixel->i, cur_perception_pix.pixel->j);
						HR_pixels.access(cur_perception_pix.pixel->i, cur_perception_pix.pixel->j).val;
				}

				/*
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
						if (offset_x == PSF_radius_x * 2 + 1) offset_x -= EX_small;
						if (offset_y == PSF_radius_y * 2 + 1) offset_y -= EX_small;

						super_offset_x = offset_x * PSF_scale;
						super_offset_y = offset_y * PSF_scale;

						LR_pixels[k][i][j].perception += super_PSF.at<double>( super_offset_y, super_offset_x ) * (HRimg.at<double>(cur_pix_i, cur_pix_j)  );
					}
					
				}
				*/

			}
		}
	}
	
}
void HRmat_to_LR_percetion ( Mat& HRimg, LR_Pixel_Array& LR_pixels, InfluenceRelation& relations/*, Mat& PSF, bool is_super_PSF, double PSF_scale*/ )
{
	for (int k = 0; k < LR_pixels.LR_imgCount; k++) {
		for (int i = 0; i < LR_pixels.LR_rows; i++) {
			for (int j = 0; j < LR_pixels.LR_cols; j++) {
				LR_Pixel& cur_LR_Pixel = LR_pixels.access(k, i, j);

				cur_LR_Pixel.perception = 0;

				for (int t = 0; t < cur_LR_Pixel.perception_link_cnt; t++) {
					Perception_Pixel& cur_perception_pix = relations.perception_links[cur_LR_Pixel.perception_link_start + t];
					cur_LR_Pixel.perception += 
						cur_perception_pix.hPSF *
						HRimg.at<uchar>(cur_perception_pix.pixel->i, cur_perception_pix.pixel->j);
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

void showConfidence_new (Mat& flow_forward, Mat& flow_backward, Mat& confidence)
{
	confidence = Mat::zeros(flow_forward.rows, flow_forward.cols, CV_64F);

	double interpScale = 10;

	Mat preInterpBackFlow;
	resize(flow_backward, preInterpBackFlow, Size(), interpScale, interpScale, INTER_CUBIC);

	for (int i = 0; i < confidence.rows; i++) for (int j = 0; j < confidence.cols; j++) {
		// forward
		Vec2f& ForwardFlow = flow_forward.at<Vec2f>(i, j);

		double tmp_posy = i + ForwardFlow[1];
		double tmp_posx = j + ForwardFlow[0];
		
		if (tmp_posy >= 0 && tmp_posx >= 0 && tmp_posy < flow_forward.rows && tmp_posx < flow_forward.cols) {
			Vec2f& BackFlow = preInterpBackFlow.at<Vec2f>((int) (tmp_posy*interpScale), (int) (tmp_posx*interpScale));
			confidence.at<double>(i, j) = calcConfidence(ForwardFlow, BackFlow);
		}
	}
}

void showConfidence (Mat& flow_forward, Mat& flow_backward, Mat& confidence, double sigmaScaler)
{
	// use CV_64FC2

	confidence = Mat::zeros(flow_forward.rows, flow_forward.cols, CV_64F);
	double sigma = SQR(sigmaScaler) * 0.4343;

	int i, j;

	Mat forw_pos = Mat::zeros(confidence.rows, confidence.cols, CV_64FC2);

	for (i = 0; i < confidence.rows; i++) for (j = 0; j < confidence.cols; j++) {
		// forward
		Vec2f& tmp_pos = forw_pos.at<Vec2f>(i, j);
		Vec2f& tmp_flow= flow_forward.at<Vec2f>(i, j);

		tmp_pos[0] = i + tmp_flow[1];
		tmp_pos[1] = j + tmp_flow[0];
	}

	double max_confdence = 0;

	double portion_i, portion_j;
	int i_to, j_to;
	int n, m;
	double sum;

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

				Vec2f interpFlow = 
					portion_i * ij + 
					(1-portion_i) * i1j;

				confidence.at<double>(i, j) = calcConfidence(cur_flow, interpFlow, sigma);
				/*
				confidence.at<double>(i, j) = 
					portion_i * calcConfidence(cur_flow, ij, sigma) + 
					(1-portion_i) * calcConfidence(cur_flow, i1j, sigma);
					/**/
			}
			else if (j_to+1 < confidence.cols) {
				// i_to == confidence.rows - 1

				portion_j = 1 - (tmp_forwpos[1] - j_to);
				Vec2f& ij = flow_backward.at<Vec2f> (i_to, j_to);
				Vec2f& ij1 = flow_backward.at<Vec2f> (i_to, j_to+1);

				Vec2f interpFlow = 
					portion_j * ij + 
					(1-portion_j) * ij1;

				confidence.at<double>(i, j) = calcConfidence(cur_flow, interpFlow, sigma);

				/*
				confidence.at<double>(i, j) = 
					portion_j * calcConfidence(cur_flow, ij, sigma) + 
					(1-portion_j) * calcConfidence(cur_flow, ij1, sigma);
					/**/
			}
			else {
				// i_to == confidence.rows - 1 && j_to == confidence.cols -1
				Vec2f& ij = flow_backward.at<Vec2f> (i_to, j_to);

				confidence.at<double>(i, j) = 
					calcConfidence(cur_flow, ij, sigma);
			}


			continue;
		}

		portion_i = 1 - (tmp_forwpos[0] - i_to);
		portion_j = 1 - (tmp_forwpos[1] - j_to);

		Vec2f& ij = flow_backward.at<Vec2f> (i_to, j_to);
		Vec2f& i1j = flow_backward.at<Vec2f> (i_to+1, j_to);
		Vec2f& ij1 = flow_backward.at<Vec2f> (i_to, j_to+1);
		Vec2f& i1j1 = flow_backward.at<Vec2f> (i_to+1, j_to+1);

		Vec2f interpFlow = 
			portion_i * portion_j * ij + 
			(1-portion_i) * portion_j * i1j + 
			portion_i * (1-portion_j) * ij1 +
			(1-portion_i) * (1-portion_j) * i1j1; 

		confidence.at<double>(i, j) = calcConfidence(cur_flow, interpFlow, sigma);

		/*
		confidence.at<double>(i, j) = 
			portion_i * portion_j * calcConfidence(cur_flow, ij, sigma) + 
			(1-portion_i) * portion_j * calcConfidence(cur_flow, i1j, sigma) + 
			portion_i * (1-portion_j) * calcConfidence(cur_flow, ij1, sigma) +
			(1-portion_i) * (1-portion_j) * calcConfidence(cur_flow, i1j1, sigma); 
		/**/
		// it's no longer useful, I guess max_confidence = 1
		if (confidence.at<double>(i, j) > max_confdence) max_confdence = confidence.at<double>(i, j);
	}

	confidence = confidence /** 254*/ + EX_small;
	//confidence = 1;
	//confidence = EX_small;
}

double ExpNegSQR (double x, double y, double sigma) {
	//double sigma = 0.4343; // make one pixel far decay to 0.1
	//double sigma = 0.2171; // make one pixel far decay to 0.01
	//double sigma = 0.1448; // make one pixel far decay to 0.001

	return exp(-(SQR(x) + SQR(y)) / sigma);
	//return (double) ((SQR(x) < 1)&&(SQR(y) < 1));
}

double calcConfidence (Vec2f& f, Vec2f& b, double sigma)
{
	double diff_x, diff_y;
	diff_x = f[0] + b[0];
	diff_y = f[1] + b[1];

	return ExpNegSQR(diff_x, diff_y, sigma);
}

void ImgPreProcess (vector<Mat>& LR_imgs, vector<Mat>& output)
{
	vector<Mat> tmps, Gx, Gy;
	tmps.resize(LR_imgs.size());
	Gx.resize(LR_imgs.size());
	Gy.resize(LR_imgs.size());

	output.resize(LR_imgs.size());

	double tmp_max = 0;
	for (int k = 0; k < LR_imgs.size(); k++) {
		tmp_max = 0;

		GaussianBlur( LR_imgs[k], tmps[k], Size(3,3), 0.01, 0.01, BORDER_DEFAULT );
		Sobel( LR_imgs[k], Gx[k], CV_64F, 1, 0, 3, 1, 0, BORDER_DEFAULT );
		Sobel( LR_imgs[k], Gy[k], CV_64F, 0, 1, 3, 1, 0, BORDER_DEFAULT );

		output[k] = Mat::zeros(LR_imgs[0].rows, LR_imgs[0].cols, CV_64F);
		for (int i = 0; i < LR_imgs[0].rows; i++) for (int j = 0; j < LR_imgs[0].cols; j++) {
			output[k].at<double>(i,j) = sqrt( SQR(Gx[k].at<double>(i,j)) + SQR(Gy[k].at<double>(i,j)) ) / ( (double) tmps[k].at<uchar>(i,j) + 1 );
			if (output[k].at<double>(i,j) > tmp_max) {
				tmp_max = output[k].at<double>(i,j);
			}
		}
	}
	
	for (int k = 0; k < LR_imgs.size(); k++) {
		output[k].convertTo(output[k], CV_8U, 255.0 / tmp_max, 0);

		//imwrite("output/" + test_set + "256_0" + int2str(k+1) + "_preProcess.bmp", output[k]);
	}
	/**/
}

void getBetterFlow (vector<Mat>& oriConfs, vector<Mat>& oriFlows, vector<Mat>& newConfs, vector<Mat>& newFlows, vector<Mat>& combinedConfs, vector<Mat>& combinedFlows)
{
	combinedFlows.resize(oriConfs.size());
	combinedConfs.resize(oriConfs.size());

	for (int k = 0; k < oriConfs.size(); k++) {
		combinedFlows[k] = Mat::zeros(oriFlows[k].rows, oriFlows[k].cols, CV_64FC2);
		combinedConfs[k] = Mat::zeros(oriFlows[k].rows, oriFlows[k].cols, CV_64F);

		for (int i = 0; i < oriConfs[0].rows; i++) for (int j = 0; j < oriConfs[0].cols; j++) {
			Vec2f& comFlow = combinedFlows[k].at<Vec2f>(i, j);

			if (oriConfs[k].at<double>(i, j) > newConfs[k].at<double>(i, j)) {
				combinedConfs[k].at<double>(i, j) = oriConfs[k].at<double>(i, j);

				Vec2f& chosenFlow = oriFlows[k].at<Vec2f>(i, j);

				comFlow[0] = chosenFlow[0];
				comFlow[1] = chosenFlow[1];
			}
			else {
				combinedConfs[k].at<double>(i, j) = newConfs[k].at<double>(i, j);

				Vec2f& chosenFlow = newFlows[k].at<Vec2f>(i, j);

				comFlow[0] = chosenFlow[0];
				comFlow[1] = chosenFlow[1];
			}
		}
	}
}

// FlexISP
/*
void formResampleMatrix (vector < vector < vector <LR_Pixel> > >& LR_pixels,
							  vector < vector <HR_Pixel> >&  HR_pixels,
							  vector <MySparseMat>& S,
							  vector <MySparseMat>& ST,
							  vector <EigenSpMat>& S_eigen,
							  vector <EigenSpMat>& ST_eigen) {
	cout << "formResampleMatrix" << endl;

 	int LR_ImgCount = LR_pixels.size(),
		LR_Rows = LR_pixels[0].size(),
		LR_Cols = LR_pixels[0][0].size(),
		HR_Rows = HR_pixels.size(),
		HR_Cols = HR_pixels[0].size();
	
	S.resize(LR_ImgCount);
	ST.resize(LR_ImgCount);
	S_eigen.resize(LR_ImgCount);
	ST_eigen.resize(LR_ImgCount);

	int size[2];
	size[0] = LR_Rows*LR_Cols, size[1] = HR_Rows*HR_Cols;

	int cur_Row, sourcePos2ColIdx, tmp_idx[2];

	vector<T> tripletList, tripletListT;

	for (int k = 0; k < S.size(); k++) {
		//tripletList.reserve(6553600);
		tripletListT.reserve(6553600);

		S[k] = MySparseMat(size[0], size[1], 1);
		ST[k] = MySparseMat(size[1], size[0], 0);
		S_eigen[k] = EigenSpMat(size[0], size[1]);
		ST_eigen[k] = EigenSpMat(size[1], size[0]);

		for (int ii = 0; ii < LR_Rows; ii++) for (int jj = 0; jj < LR_Cols; jj++) {
			cur_Row = ii * LR_Cols + jj;
			tmp_idx[0] = cur_Row;

			for (int p = 0; p < LR_pixels[k][ii][jj].perception_pixels.size(); p++) {
				sourcePos2ColIdx = LR_pixels[k][ii][jj].perception_pixels[p].pixel -> i * HR_Cols +
					LR_pixels[k][ii][jj].perception_pixels[p].pixel -> j;

				tmp_idx[1] = sourcePos2ColIdx;

				S[k].setVal( cur_Row, sourcePos2ColIdx, LR_pixels[k][ii][jj].perception_pixels[p].hPSF );
				ST[k].setVal( cur_Row, sourcePos2ColIdx, LR_pixels[k][ii][jj].perception_pixels[p].hPSF );

				//tripletList.push_back(T(cur_Row, sourcePos2ColIdx, LR_pixels[k][ii][jj].perception_pixels[p].hPSF));	
				tripletListT.push_back(T(sourcePos2ColIdx, cur_Row, LR_pixels[k][ii][jj].perception_pixels[p].hPSF));	
			}
			
		}
		//S[k].setFromTriplets(tripletList.begin(), tripletList.end());
		ST_eigen[k].setFromTriplets(tripletListT.begin(), tripletListT.end());
		//tripletList.clear();
		tripletListT.clear();
	}
}
*/
/*
void resampleByMatrix (Mat& X,
					   vector <EigenSpRowMat>& S, 
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
			for (EigenSpRowMat::InnerIterator it(S[k],tk); it; ++it)
			{
				LR_idx[0] = it.row() / LR_Cols;
				LR_idx[1] = it.row() % LR_Cols;
				
				HR_idx[0] = it.col() / HR_Cols;
				HR_idx[1] = it.col() % HR_Cols;

				SX[k].at<double>(LR_idx[0], LR_idx[1]) += it.value() * X.at<double>(HR_idx[0], HR_idx[1]);
			}
	}
}
*/
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

void multiplyMySpMat (MySparseMat& A, MySparseMat& B, EigenSpMat& out) {
	cout << "multiplyMySpMat...\n";

	if (A.type != 0 || B.type != 1) {
		cout << "wrong type of A or B\n";

		return ;
	}

	out = EigenSpMat(A.rows, B.cols);
	vector<T> tripletList;
	tripletList.reserve(6553600);

	cout << A.elements.size() << endl;
	cout << B.elements.size() << endl;

	for (int t = 0; t < A.elements.size(); t++) {
		cout << "\rmultiplying: " << t << "th row";
		for (int s = 0; s < B.elements.size(); s++) {
			if (t == 56) {
				cout << "\n" << s << " th col";
			}
			//cout << "\rmultiplying " << t << "th row & " << s << " th col";
			tripletList.push_back(T(t, s, MySpMat_dot(A.elements[t], B.elements[s])));
		}
	}

	out.setFromTriplets(tripletList.begin(), tripletList.end());
}

double MySpMat_dot (vector<Element>& a, vector<Element>& b) {
	double out = 0;

	for (int t = 0; t < a.size(); t++) {
		for (int s = 0; s < b.size(); s++) {
			if (a[t].j == b[s].i) {
				out += a[t].val * b[s].val;
			}
		}
	}

	return out;
}

void DivideToBlocksToConstruct(vector<Mat>& BigLRimgs, vector<Mat>& BigFlows, vector<Mat>& BigConfs, Mat& PSF, double scale, Mat& BigHRimg)
{
	int overlappingPix = 5;

	//get the image data
	int BigLR_rows = BigLRimgs[0].rows;
	int BigLR_cols = BigLRimgs[0].cols;
	int BigHR_rows = BigLR_rows * scale;
	int BigHR_cols = BigLR_cols * scale;

	int LR_imgCount = BigLRimgs.size();
	BigHRimg = Mat::zeros(BigHR_rows, BigHR_cols, CV_64F);

	int longSide = (BigHR_rows > BigHR_cols) ? BigHR_rows : BigHR_cols;
	int totalBlocksCount = pow(4, floor(log(longSide/200.f)/log(2.0f))); // origin: ceil

	double blockPerAxis = sqrt(totalBlocksCount);
	cout << endl << "blockPerAxis: " << blockPerAxis << endl;
	double blockWidth = double(BigLR_cols)/blockPerAxis;
	double blockHeight = double(BigLR_rows)/blockPerAxis;
	int blockCount = 0;

	for( double y = 0; y < BigLR_rows; ) {
		int imgx, imgy, imgwidth, imgheight, imgOriginWidth, imgOriginHeight;
		int overlappingx, overlappingy, overlappingw, overlappingh;
		for( double x =  0 ; x < BigLR_cols ; ) {
			cout << "Block (" << y << ", " << x << ") of (" << BigLR_rows << ", " << BigLR_cols << ")\n";

			imgx = int(x);
			imgy = int(y);
			imgwidth = floor(x+blockWidth+0.5+EX_small) - imgx;
			imgheight = floor(y+blockHeight+0.5+EX_small) - imgy;

			imgOriginWidth = imgwidth;
			imgOriginHeight = imgheight;

			imgx = imgx-overlappingPix;
			if(imgx < 0) { imgx = 0; overlappingx = 0; } else { overlappingx = overlappingPix; }
			imgy = imgy-overlappingPix;
			if(imgy < 0) { imgy = 0; overlappingy = 0; } else { overlappingy = overlappingPix; }
			imgwidth = imgwidth+overlappingPix+overlappingx;  //because imgx had minus overlapping, so need to add *2 to overlap
			if((imgx+imgwidth)>=BigLR_cols) { imgwidth = BigLR_cols-imgx; overlappingw = 0; } else { overlappingw = overlappingPix; }
			imgheight = imgheight+overlappingPix+overlappingy;
			if((imgy+imgheight)>=BigLR_rows) { imgheight = BigLR_rows-imgy; overlappingh = 0; } else { overlappingh = overlappingPix; }

			Rect rect = Rect ( imgx, imgy , imgwidth, imgheight );

			vector<Mat> imagesBlock;
			vector<Mat> flowsBlock;
			vector<Mat> confsBlock;
			imagesBlock.resize(LR_imgCount);
			flowsBlock.resize(LR_imgCount);
			confsBlock.resize(LR_imgCount);
			for (int k = 0; k < LR_imgCount; k++) {
				Mat img = BigLRimgs[k];
				Mat imgBlock = Mat(img, rect);
				imagesBlock[k] = imgBlock.clone();

				Mat flow = BigFlows[k];
				Mat flowBlock = Mat(flow, rect);
				flowsBlock[k] = flowBlock.clone();

				Mat conf = BigConfs[k];
				Mat confBlock = Mat(conf, rect);
				confsBlock[k] = confBlock.clone();
			}
			Mat SmallHRimg;
			// ----- CONSTRUCTION -----
			Linear_Constructor linearConstructor( imagesBlock, flowsBlock, confsBlock, scale, PSF);
			linearConstructor.addRegularization_grad2norm(0.05);
			linearConstructor.solve_byCG();
			linearConstructor.output(SmallHRimg);

			cout << "block construct complete\n";
			// ----- CONSTRUCTION -----

			int rowst = imgy+overlappingy, rowlength = imgheight-overlappingh-overlappingy, colst = imgx+overlappingx, collength = imgwidth-overlappingw-overlappingx;
			//Rect rectInHR = Rect( colst*2, rowst*2, collength*2, rowlength*2 );
			Rect rectInHR = Rect( colst*scale, rowst*scale, collength*scale, rowlength*scale );
			cout << "rectInHR: " << rectInHR << endl;
			//Rect rectInHRBlock = Rect( (colst-imgx)*2, (rowst-imgy)*2, collength*2, rowlength*2 );
			Rect rectInHRBlock = Rect( (colst-imgx)*scale, (rowst-imgy)*scale, collength*scale, rowlength*scale );
			cout << "rectInHRBlock: " << rectInHRBlock << endl;
			//cout << rect << endl << rectInHR << endl << rectInHRBlock << endl << HRSingleImg.size()  << endl << HRSingleImgFinal.size() << endl;
			//HRSingleImg(rectInHRBlock).copyTo(HRSingleImgFinal(rectInHR));
			cout << "SmallHRimg.size(): " << SmallHRimg.size() << endl;
			SmallHRimg(rectInHRBlock).copyTo(BigHRimg(rectInHR));
			cout << "gg\n";

			blockCount++;
			if(imgOriginWidth > blockWidth) x = imgx+overlappingx+imgOriginWidth;  //the fraction bigger than 0.5, reset to no fraction
			else x = x+blockWidth;  //keep fraction
		}
		if(imgOriginHeight > blockHeight) y = imgy+overlappingy+imgOriginHeight;
		else y = y+blockHeight;
	}

	
}

//

//void weightedNeighborWarp (vector<vector<HR_Pixel> >& HR_pixels, Mat& HRimg)
//{
//	HRimg = Mat::zeros(HR_pixels.size(), HR_pixels[0].size(), CV_64F);
//
//	int i, j, k;
//	
//	double tmp_sum, tmp_d_sum;
//	for (i = 0; i < HR_pixels.size(); i++) {
//		for (j = 0; j < HR_pixels[i].size(); j++) {
//			//HRimg.at<double>(i,j) = (double)tmp_HR.at<uchar>(i,j);
//			
//			tmp_sum = 0;
//			tmp_d_sum = 0;
//			for (k = 0; k < HR_pixels[i][j].influenced_pixels.size(); k++) {
//				tmp_sum += HR_pixels[i][j].influenced_pixels[k].pixel->val * HR_pixels[i][j].influenced_pixels[k].hBP * HR_pixels[i][j].influenced_pixels[k].pixel->confidence/**/;
//				tmp_d_sum += HR_pixels[i][j].influenced_pixels[k].hBP /** influence_bucket[i][j].influenced_pixels[k].pixel->confidence/**/;
//			}
//			
//			if (HR_pixels[i][j].influenced_pixels.size() == 0) {
//				HRimg.at<double>(i,j) = tmp_sum;
//			}
//			else {
//				//tmp_d_sum = HR_pixels[i][j].influenced_pixels.size();
//				HRimg.at<double>(i,j) = tmp_sum / tmp_d_sum;
//			}
//			/**/
//		}
//	}
//
//}

void warpImageByFlow (Mat& colorImg, Mat& flow, Mat& output) {

	Mat map_x(flow.size(), CV_32FC1);
	Mat map_y(flow.size(), CV_32FC1);
	for (int i = 0; i < map_x.rows; i++)
	{
		for (int j = 0; j < map_x.cols; j++)
		{
			Vec2f f = flow.at<Vec2f>(i, j);
			map_x.at<float>(i, j) = j + f.val[0];
			map_y.at<float>(i, j) = i + f.val[1];
		}
	}

	remap(colorImg, output, map_x, map_y, CV_INTER_CUBIC);

}

void outputHRcolor (Mat& HRimgC1, Mat& LRimg, Mat& HRimgC3) {

	Mat tmp, tmp2;
	cvtColor(LRimg, tmp, CV_BGR2YCrCb);
	resize(tmp, tmp2, HRimgC1.size(), 0, 0, CV_INTER_CUBIC);

	if (HRimgC1.type() == CV_64F) {
		for (int i = 0; i < tmp2.rows; i++) for (int j = 0; j < tmp2.cols; j++)
		{
			tmp2.at<Vec3b>(i, j)[0] = saturate_cast<uchar>(HRimgC1.at<double>(i, j));
		}
	}
	else if (HRimgC1.type() == CV_8U) {
		for (int i = 0; i < tmp2.rows; i++) for (int j = 0; j < tmp2.cols; j++)
		{
			tmp2.at<Vec3b>(i, j)[0] = HRimgC1.at<uchar>(i, j);
		}
	}

	cvtColor(tmp2, HRimgC3, CV_YCrCb2BGR);

}

void specialBlur (Mat& input, Mat& output) {
	Mat tmp, tmp2;
	Mat peak, cut;

	Laplacian(input, tmp, CV_64F, 1, 1, 0, BORDER_REPLICATE );
	peak = Mat::zeros(input.size(), CV_64F);
	peak = tmp * (-0.25);
	imwrite("output/peak.bmp", peak);
	subtract(input, peak, cut, noArray(), CV_64F);
	imwrite("output/cut.bmp", cut);
	//cut = input - peak;
	cut.convertTo(output, CV_8U);

	/*
	GaussianBlur(peak, tmp2, Size(0,0), 1, 1, BORDER_REPLICATE);
	cout << "nn";
	add(cut, tmp2, output, noArray(), CV_8U);
	imwrite("output/out.bmp", output);
	/**/
	//output = cut + tmp2;

}

void optFlowHS (Mat& from, Mat& to, Mat& flow, int useInit)
{
	CvMat prev = from, curr = to;

	if (useInit == 0) {
		flow = Mat::zeros(from.size(), CV_32FC2);
	}

	Mat_<float> vel[2];
	split(flow, vel);

	CvMat velx = vel[0], vely = vel[1];
	/*
	float *tmpx = (float*)malloc(from.rows * from.cols * sizeof(float)), *tmpy = (float*)malloc(from.rows * from.cols * sizeof(float));
	CvMat velx = cvMat( from.rows, from.cols, CV_32FC1,  tmpx),
		vely = cvMat( to.rows, to.cols, CV_32FC1, tmpy);//*/
	
	CvTermCriteria criteria;
	criteria.type = CV_TERMCRIT_ITER;
	criteria.max_iter = 500;
	criteria.epsilon = 0.0001;
	cvCalcOpticalFlowHS(&prev, &curr, useInit, &velx, &vely, 0.0001, criteria);

	for (int i = 0; i < flow.rows; i++) for (int j = 0; j < flow.cols; j++)
	{
		Vec2f& tmp = flow.at<Vec2f>(i, j);

		tmp.val[0] = CV_MAT_ELEM(velx, float, i, j);
		tmp.val[1] = CV_MAT_ELEM(vely, float, i, j);

	}
	/*
	free(tmpx);
	free(tmpy);
	//*/
}

void calcVecMatDiff (Mat& a, Mat& b, Mat& output)
{
	output = Mat::zeros(a.size(), CV_64F);

	for (int i = 0; i < a.rows; i++) for (int j = 0; j < a.cols; j++)
	{
		Vec2f& tmp1 = a.at<Vec2f>(i, j);
		Vec2f& tmp2 = b.at<Vec2f>(i, j);

		output.at<double>(i, j) = ExpNegSQR(tmp1[0]-tmp2[0], tmp1[1]-tmp2[1]);
	}
}