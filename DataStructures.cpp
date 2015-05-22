#include "DataStructures.h"

Pixel::Pixel()
{
	pos_x = 0;
	pos_y = 0;
	val = 0;
}

LR_Pixel::LR_Pixel()
{
	confidence = 0;
	perception = 0;
	perception_link_cnt = 0;
}

HR_Pixel::HR_Pixel()
{
	hBP_sum = 0;
	influence_link_cnt = 0;
}
//
HR_Pixel_Array::HR_Pixel_Array(int r, int c)
{
	hr_pixels = new HR_Pixel[r * c];
	HR_rows = r;
	HR_cols = c;
	HR_pixelCount = r * c;

	for (int i = 0; i < r; i++) for (int j = 0; j < c; j++) {
		hr_pixels[i * HR_cols + j].i = i;
		hr_pixels[i * HR_cols + j].j = j;
	}
}

HR_Pixel& HR_Pixel_Array::access(int i, int j)
{
	return hr_pixels[i * HR_cols + j];
}

HR_Pixel& HR_Pixel_Array::access(int idx)
{
	return hr_pixels[idx];
}

HR_Pixel_Array::~HR_Pixel_Array()
{
	delete[] hr_pixels;
}

LR_Pixel_Array::LR_Pixel_Array(int n, int r, int c)
{
	lr_pixels = new LR_Pixel[n * r * c];
	LR_imgCount = n;
	LR_rows = r;
	LR_cols = c;
	LR_pixelCount = r * c;

	for (int k = 0; k < n; k++) for (int i = 0; i < r; i++) for (int j = 0; j < c; j++) {
		lr_pixels[k * LR_pixelCount + i * LR_cols + j].k = k;
		lr_pixels[k * LR_pixelCount + i * LR_cols + j].i = i;
		lr_pixels[k * LR_pixelCount + i * LR_cols + j].j = j;
	}
}

LR_Pixel& LR_Pixel_Array::access(int k, int i, int j)
{
	return lr_pixels[k * LR_pixelCount + i * LR_cols + j];
}

LR_Pixel& LR_Pixel_Array::access(int idx)
{
	return lr_pixels[idx];
}

LR_Pixel_Array::~LR_Pixel_Array()
{
	delete[] lr_pixels;
}

// Influence Relations
bool compare_hr_idx (Influenced_Pixel& a,Influenced_Pixel& b) { 
	return (a.hr_idx < b.hr_idx); 
}

bool compare_lr_idx (Perception_Pixel& a,Perception_Pixel& b) { 
	return (a.lr_idx < b.lr_idx); 
}

InfluenceRelation::InfluenceRelation(vector<Mat>& imgs,
							vector<Mat>& flows,
							LR_Pixel_Array* LR_pixels,
							HR_Pixel_Array*  HR_pixels,
							double scale,
							Mat& super_PSF,
							Mat& super_BPk,
							double interp_scale)
{
	cout << "formInfluenceRelation(Object)" << endl;

	int i, j, k;
	int x, y;
	
	int LR_rows = imgs[0].rows;
	int LR_cols = imgs[0].cols;
	int HR_rows = LR_rows * scale;
	int HR_cols = LR_cols * scale;
	int LR_imgCount = imgs.size();
	
	int PSF_radius_x = super_PSF.cols / interp_scale / 2;
	int PSF_radius_y = super_PSF.rows / interp_scale / 2;
	int BPk_radius_x = super_BPk.cols / interp_scale / 2;
	int BPk_radius_y = super_BPk.rows / interp_scale / 2;

	double pos_x, pos_y, bucket_center_x, bucket_center_y, dist_x, dist_y, dx, dy, offset_x, offset_y;;
	int bucket_idx_i, bucket_idx_j, super_offset_x, super_offset_y;
	
	// start record
	// for each image
	for (k = 0; k < LR_imgCount; k++) {
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
						if (offset_x == BPk_radius_x * 2 + 1) offset_x -= EXsmall;
						if (offset_y == BPk_radius_y * 2 + 1) offset_y -= EXsmall;
						super_offset_x = offset_x * interp_scale;
						super_offset_y = offset_y * interp_scale;
						tmp_pix.hBP = super_BPk.at<double>(super_offset_x, super_offset_y);
						// add to bucket
						HR_pixels->access( bucket_idx_i, bucket_idx_j).hBP_sum += tmp_pix.hBP;
						// we now save all influenced_pixels to influence_links
						//HR_pixels->access( bucket_idx_i, bucket_idx_j).influenced_pixels.push_back( tmp_pix );
						tmp_pix.hr_idx = (bucket_idx_i * HR_cols + bucket_idx_j);
						tmp_pix.lr_idx = (k * LR_pixels->LR_pixelCount + i * LR_cols + j);
						influence_links.push_back( tmp_pix );						

						// create a perception relation
						Perception_Pixel tmp_pix2;
						tmp_pix2.pixel = &(HR_pixels->access( bucket_idx_i, bucket_idx_j));
						// ----- hpsf
						offset_x = (dx) + PSF_radius_x + 0.5;
						offset_y = (dy) + PSF_radius_y + 0.5;
						if (offset_x == PSF_radius_x * 2 + 1) offset_x -= EXsmall;
						if (offset_y == PSF_radius_y * 2 + 1) offset_y -= EXsmall;
						super_offset_x = offset_x * interp_scale;
						super_offset_y = offset_y * interp_scale;
						tmp_pix2.hPSF = super_PSF.at<double>(super_offset_x, super_offset_y);

						// we now save all perception_pixels to perception_links
						//LR_pixels->access(k, i, j).perception_pixels.push_back(tmp_pix2);
						tmp_pix2.hr_idx = (bucket_idx_i * HR_cols + bucket_idx_j);
						tmp_pix2.lr_idx = (k * LR_pixels->LR_pixelCount + i * LR_cols + j);
						perception_links.push_back( tmp_pix2 );
					}
				}
			}
		}
	}

	// start sort links
	sort (influence_links.begin(), influence_links.end(), compare_hr_idx);
	sort (perception_links.begin(), perception_links.end(), compare_lr_idx);

	// influence_links assign to hr_pixels
	int cur_hr_idx = -1;
	for (int idx = 0; idx < influence_links.size(); idx++) {
		if (cur_hr_idx != influence_links[idx].hr_idx) {
			cur_hr_idx = influence_links[idx].hr_idx;
			HR_pixels->access(cur_hr_idx).influence_link_start = idx;
			HR_pixels->access(cur_hr_idx).influence_link_cnt++;
		}
		else {
			HR_pixels->access(cur_hr_idx).influence_link_cnt++;
		}
	}

	// perception_links assign to lr_pixels
	int cur_lr_idx = -1;
	for (int idx = 0; idx < perception_links.size(); idx++) {
		if (cur_lr_idx != perception_links[idx].lr_idx) {
			cur_lr_idx = perception_links[idx].lr_idx;
			LR_pixels->access(cur_lr_idx).perception_link_start = idx;
			LR_pixels->access(cur_lr_idx).perception_link_cnt++;
		}
		else {
			LR_pixels->access(cur_lr_idx).perception_link_cnt++;
		}
	}

}

//-----
MySparseMat::MySparseMat()
{
	type = 0;
	nzcount = 0;
	rows = 0;
	cols = 0;
}

MySparseMat::MySparseMat(int r, int c, int t)
{
	type = t;
	nzcount = 0;
	rows = r;
	cols = c;

	if (type == 0) {
		elements.resize(r);
	}
	else {
		elements.resize(c);
	}
}

void MySparseMat::insertElement(Element& e)
{
	if (type == 0) {
		elements[e.i].push_back(e);
	}
	else {
		elements[e.j].push_back(e);
	}

	nzcount++;
}

void MySparseMat::setVal(int i, int j, double val)
{
	bool found = false;
	int k;

	if (type == 0) {
		for (k = 0; k < elements[i].size(); k++) {
			if (elements[i][k].j == j) {
				found = true;
				break;
			}
		}

		if (found) {
			elements[i][k].val = val;
		}
		else {
			insertElement(Element(i, j, val));
		}
	}
	else {
		for (k = 0; k < elements[j].size(); k++) {
			if (elements[j][k].i == i) {
				found = true;
				break;
			}
		}

		if (found) {
			elements[j][k].val = val;
		}
		else {
			insertElement(Element(i, j, val));
		}
	}
}

double MySparseMat::getVal(int i, int j)
{
	int k;
	bool found = false;

	if (type == 0) {
		for (k = 0; k < elements[i].size(); k++) {
			if (elements[i][k].j == j) {
				found = true;
				break;
			}
		}

		if (found) {
			return elements[i][k].val;
		}
		else {
			return 0;
		}
	}
	else {
		for (k = 0; k < elements[j].size(); k++) {
			if (elements[j][k].i == i) {
				found = true;
				break;
			}
		}

		if (found) {
			return elements[j][k].val;
		}
		else {
			return 0;
		}
	}
}

Element::Element(int ii, int jj, double value)
{
	i = ii;
	j = jj;
	val = value;
}