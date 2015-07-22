#include "Linear_Reconstruction.h";

Linear_Constructor::Linear_Constructor( vector<Mat>& LR_imgs, vector<Mat>& flows, double scale, Mat& PSF) {
	cout << "----- Linear-Constructor (no Conf) -----\n";

	vector<Mat> confs;
	confs.resize(LR_imgs.size());
	confs[0] = Mat::ones(LR_imgs[0].size(), CV_64F);
	for (int i = 0; i < confs.size(); i++) 
	{
		confs[i] = confs[0];
	}

	constructor(LR_imgs, flows, confs, scale, PSF);
	//
	addDataFidelityWithConf(confs);

	cout << "----- Linear-Constructor ----- CONSTRUCT COMPLETE\n";
}
/**/

Linear_Constructor::Linear_Constructor( vector<Mat>& LR_imgs, vector<Mat>& flows, vector<Mat> confs, double scale, Mat& PSF) {
	cout << "----- Linear-Constructor -----\n";

	constructor(LR_imgs, flows, confs, scale, PSF);
	//
	addDataFidelityWithConf(confs);

	cout << "----- Linear-Constructor ----- CONSTRUCT COMPLETE\n";
}

Linear_Constructor::Linear_Constructor( DataChunk& dataChunk ) {
	cout << "----- Linear-Constructor (Block) -----\n";

	HR_rows = dataChunk.SmallHR_rows;
	HR_cols = dataChunk.SmallHR_cols;
	LR_pixelCount = dataChunk.data_LR_pix.size();
	HR_pixelCount = HR_rows * HR_cols;

	relations = dataChunk.tmp_relations;
	HR_pixels = dataChunk.tmp_HR_pixels;
	//LR_pixels = new LR_Pixel_Array(1,1,1);

	addDataFidelityWithConf(dataChunk);

	needRelease = false;

	cout << "----- Linear-Constructor ----- CONSTRUCT COMPLETE\n";
}

void Linear_Constructor::constructor( vector<Mat>& LR_imgs, vector<Mat>& flows, vector<Mat> confs, double scale, Mat& PSF)
{
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
	// initialize influenced pixels (for each pixel in each LR img)
	cout << "alloc LR_pix_array\n";
	LR_pixels = new LR_Pixel_Array(LR_imgCount, LR_rows, LR_cols);
	//
	relations = new InfluenceRelation (LR_imgs,
							flows,
							LR_pixels,
							HR_pixels,
							scale,
							super_PSF,
							super_BPk,
							interp_scale,
							confs);

	needRelease = true;
}

void Linear_Constructor::addDataFidelityWithConf(vector<Mat>& conf ) {
	cout << "add Data Fidelity Term\n";

	int sourcePos2ColIdx;
	double curConf;

	rowCnt_A = 0;

	A_triplets.reserve(relations->perception_links.size());
	b_vec.reserve(LR_imgCount*LR_pixelCount);

	for (int k = 0; k < LR_imgCount; k++) {

		for (int ii = 0; ii < LR_rows; ii++) for (int jj = 0; jj < LR_cols; jj++) {
			LR_Pixel& cur_LR_pix = LR_pixels->access(k, ii, jj);
			//curConf = conf[k].at<double>(ii, jj);
			curConf = cur_LR_pix.confidence;

			// A
			for (int p = 0; p < cur_LR_pix.perception_link_cnt; p++) {
				Perception_Pixel& cur_perception_pix = relations->perception_links[cur_LR_pix.perception_link_start + p];
				sourcePos2ColIdx = cur_perception_pix.pixel -> i * HR_cols + cur_perception_pix.pixel -> j;

				A_triplets.push_back( T(rowCnt_A, sourcePos2ColIdx, curConf * cur_perception_pix.hPSF) );	
			}
			// b
			b_vec.push_back( curConf * cur_LR_pix.val );

			// iteration update
			rowCnt_A++;
		}

	}

}

void Linear_Constructor::addDataFidelityWithConf(DataChunk& dataChunk ) {
	cout << "add Data Fidelity Term (chunk)\n";

	int sourcePos2ColIdx;
	double curConf;

	rowCnt_A = 0;

	//A_triplets.reserve(relations->perception_links.size());
	//b_vec.reserve(LR_imgCount*LR_pixelCount);

	for (int idx = 0; idx < dataChunk.data_LR_pix.size(); idx ++) {
		LR_Pixel& cur_LR_pix = dataChunk.data_LR_pix[idx];
		curConf = cur_LR_pix.confidence;

		// A
		for (int p = 0; p < cur_LR_pix.perception_link_cnt; p++) {
			Perception_Pixel& cur_perception_pix = relations->perception_links[cur_LR_pix.perception_link_start + p];
			sourcePos2ColIdx = cur_perception_pix.pixel -> i * HR_cols + cur_perception_pix.pixel -> j;
			
			A_triplets.push_back( T(rowCnt_A, sourcePos2ColIdx, curConf * cur_perception_pix.hPSF) );	
		}
		// b
		b_vec.push_back( curConf * cur_LR_pix.val );
		
		// iteration update
		rowCnt_A++;

	}
}

void Linear_Constructor::addRegularization_grad2norm(double gamma) {
	cout << "adding Regularization: Gradient 2 norm\n";

	int HR2ColIdx, cur_HR_idx;
	double sqrtGamma = sqrt(gamma);

	rowCnt_C = 0;
	C_triplets.reserve(4 * HR_pixelCount);
	//b_vec.reserve(b_vec.size() + 2 * HR_pixelCount);

	// Grad x
	for (int i = 0; i < HR_rows; i++) for (int j = 0; j < HR_cols - 1; j++) {
		cur_HR_idx = i * HR_cols + j;

		C_triplets.push_back( T(rowCnt_C, cur_HR_idx, sqrtGamma) );
		C_triplets.push_back( T(rowCnt_C, cur_HR_idx+1, -sqrtGamma) );
		//b_vec.push_back(0);

		cur_HR_idx ++;
		rowCnt_C ++;
	}
	// Grad y
	for (int i = 0; i < HR_rows - 1; i++) for (int j = 0; j < HR_cols; j++) {
		cur_HR_idx = i * HR_cols + j;

		C_triplets.push_back( T(rowCnt_C, cur_HR_idx, sqrtGamma) );
		C_triplets.push_back( T(rowCnt_C, cur_HR_idx+HR_cols, -sqrtGamma) );
		//b_vec.push_back(0);

		cur_HR_idx ++;
		rowCnt_C ++;
	}

}

void Linear_Constructor::solve_by_CG() {
	
	int i;

	cout << "solve by CG\n";

	A = EigenSpMat( rowCnt_A + rowCnt_C, HR_pixelCount );
	b = VectorXd( rowCnt_A + rowCnt_C );
	x = VectorXd( HR_pixelCount );

	cout << "forming A ...\n";
	for (i = 0; i < C_triplets.size(); i++) {
		A_triplets.push_back(T(C_triplets[i].row() + rowCnt_A, C_triplets[i].col(), C_triplets[i].value()));
	}
	A.setFromTriplets( A_triplets.begin(), A_triplets.end() );
	cout << "A.size(): " << A.rows() << ", " << A.cols() << endl;
	//A.makeCompressed();

	cout << "forming b ...\n";

	for (i = 0; i < rowCnt_A; i++) {
		b(i) = b_vec[i];
	}
	for (; i < rowCnt_A + rowCnt_C; i++) {
		b(i) = 0;
	}

	
	cout << "forming AT...\n";
	EigenSpMat AT = A.transpose();
	//AT.makeCompressed();
	
	cout << "multiplying ATA..." << endl;
	ATA = (AT * A).pruned(100, EX_small);
	//ATA.makeCompressed();
	cout << "multiplying ATb...\n";
	ATb = AT * b;

	cout << "construct solver\n";
	ConjugateGradient<EigenSpMat> CG_sover(ATA);
	cout << "solving...\n";
	x = CG_sover.solve(ATb);
}

void Linear_Constructor::solve_by_GradientDescent()
{
	int i;

	cout << "solve by Gradient Descent\n";

	A = EigenSpMat( rowCnt_A, HR_pixelCount );
	b = VectorXd( rowCnt_A);
	C = EigenSpMat( rowCnt_C, HR_pixelCount );
	x_n = VectorXd( HR_pixelCount );
	x_n1 = VectorXd( HR_pixelCount );

	cout << "forming A ...\n";
	A.setFromTriplets( A_triplets.begin(), A_triplets.end() );
	cout << "A.size(): " << A.rows() << ", " << A.cols() << endl;

	cout << "forming b ...\n";
	for (i = 0; i < rowCnt_A; i++) {
		b(i) = b_vec[i];
	}

	cout << "forming x (initial guess) ...\n";
	for (i = 0; i < HR_pixelCount; i++) {
		x_n(i) = 0;
		x_n1(i) = 0;
	}

	cout << "forming C ...\n";
	C.setFromTriplets( C_triplets.begin(), C_triplets.end() );
	cout << "C.size(): " << C.rows() << ", " << C.cols() << endl;

	cout << "forming AT...\n";
	EigenSpMat AT = A.transpose();

	cout << "forming CT...\n";
	EigenSpMat CT = C.transpose();

	cout << "multiplying ATA..." << endl;
	ATA = (AT * A).pruned(100, EX_small);

	cout << "multiplying CTC..." << endl;
	CTC = (CT * C).pruned(100, EX_small);

	cout << "multiplying ATb...\n";
	ATb = AT * b;

	double err = EX_big;
	// 0.0001
	while (err > 0.001) {
		x_n1 = x_n + 0.5*(ATb - ATA * x_n - CTC * x_n);

		err = (x_n1 - x_n).norm() / HR_pixelCount;
		//cout << "err: " << err << endl;

		x_n = x_n1;
	}

	x = x_n1;
}

void Linear_Constructor::output(Mat& HRimg) {
	cout << "output as Mat\n";
	HRimg = Mat::zeros(HR_rows, HR_cols, CV_64F);
	int curIdx = 0;
	for (int i = 0; i < HR_rows; i++) for (int j = 0; j < HR_cols; j++) {
		HRimg.at<double>(i, j) = x(curIdx);

		curIdx ++;
	}
}

Linear_Constructor::~Linear_Constructor()
{
	if (needRelease){
		delete HR_pixels;
		delete LR_pixels;
		delete relations;
	}	
}
//*/