#include "Linear_Construction.h";

LinearConstructor::LinearConstructor( vector<Mat>& LR_imgs, vector<Mat>& flows, double scale, Mat& PSF) {
	cout << "----- Linear-Constructor -----\n";

	int i, j, k;

	interp_scale = 1000;

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
	HR_pixels.resize(HR_rows);
	for (i = 0; i < HR_pixels.size(); i++) {
		HR_pixels[i].resize(HR_cols);
	}
	for (i = 0; i < HR_pixels.size(); i++) for (j = 0; j < HR_pixels[0].size(); j++) {
		HR_pixels[i][j].i = i;
		HR_pixels[i][j].j = j;
	}
	// initialize influenced pixels (for each pixel in each LR img)
	LR_pixels.resize(LR_imgCount);
	for (k = 0; k < LR_imgCount; k++) {
		LR_pixels[k].resize(LR_rows);
		for (i = 0; i < LR_rows; i++) {
			LR_pixels[k][i].resize(LR_cols);
		}
	}
	for (k = 0; k < LR_imgCount; k++) for (i = 0; i < LR_rows; i++) for (j = 0; j < LR_cols; j++) {
		LR_pixels[k][i][j].i = i;
		LR_pixels[k][i][j].j = j;
		LR_pixels[k][i][j].k = k;		
	}
	//
	formInfluenceRelation (
		LR_imgs,
		flows,
		LR_pixels,
		HR_pixels,
		scale,
		super_PSF,
		super_BPk,
		interp_scale
		);
	//
	A_triplets.reserve( HR_pixelCount * (PSF.rows+1) * (PSF.cols+1) * LR_imgCount + 5 * HR_pixelCount );
	b_vec.reserve( LR_imgCount * LR_pixelCount + 5 * HR_pixelCount );
	curRow = 0;
	addDataFidelityTerm();

	cout << "----- Linear-Constructor ----- CONSTRUCT COMPLETE\n";
}

void LinearConstructor::addDataFidelityTerm( ) {
	cout << "add Data Fidelity Term\n";

	int sourcePos2ColIdx;

	for (int k = 0; k < LR_imgCount; k++) {

		for (int ii = 0; ii < LR_rows; ii++) for (int jj = 0; jj < LR_cols; jj++) {
			LR_Pixel& cur_LR_pix = LR_pixels[k][ii][jj];

			// A
			for (int p = 0; p < cur_LR_pix.perception_pixels.size(); p++) {

				sourcePos2ColIdx = cur_LR_pix.perception_pixels[p].pixel -> i * HR_cols +	cur_LR_pix.perception_pixels[p].pixel -> j;

				A_triplets.push_back( T(curRow, sourcePos2ColIdx, cur_LR_pix.perception_pixels[p].hPSF) );	
			}
			// b
			b_vec.push_back( cur_LR_pix.val );

			// iteration update
			curRow++;
		}

	}

}

void LinearConstructor::addRegularization_grad2norm() {
	cout << "adding Regularization: Gradient 2 norm\n";

	int HR2ColIdx, cur_HR_idx;

	// Grad x
	for (int i = 0; i < HR_rows; i++) for (int j = 0; j < HR_cols - 1; j++) {
		cur_HR_idx = i * HR_cols + j;

		A_triplets.push_back( T(curRow, cur_HR_idx, 1.0) );
		A_triplets.push_back( T(curRow, cur_HR_idx+1, -1.0) );
		b_vec.push_back(0);

		cur_HR_idx ++;
		curRow ++;
	}
	// Grad y
	for (int i = 0; i < HR_rows - 1; i++) for (int j = 0; j < HR_cols; j++) {
		cur_HR_idx = i * HR_cols + j;

		A_triplets.push_back( T(curRow, cur_HR_idx, 1.0) );
		A_triplets.push_back( T(curRow, cur_HR_idx+HR_cols, -1.0) );
		b_vec.push_back(0);

		cur_HR_idx ++;
		curRow ++;
	}

}

void LinearConstructor::solve_bySparseQR() {
	cout << "solve by SparseQR\n";

	A = EigenSpMat( curRow, HR_pixelCount );
	b = VectorXd( curRow );
	x = VectorXd( curRow );

	cout << "forming A ...\n";
	A.setFromTriplets( A_triplets.begin(), A_triplets.end() );
	A.makeCompressed();

	cout << "forming b ...\n";
	for (int i = 0; i < curRow; i++) {
		b(i) = b_vec[i];
	}

	cout << "construct solver\n";
	SparseQR<EigenSpMat, COLAMDOrdering<int> > SparseQR_solver(A);
	cout << "solving...\n";
	x = SparseQR_solver.solve(b);
}

void LinearConstructor::solve_byCG() {
	cout << "solve by CG\n";

	A = EigenSpMat( curRow, HR_pixelCount );
	b = VectorXd( curRow );
	x = VectorXd( curRow );

	cout << "forming A ...\n";
	A.setFromTriplets( A_triplets.begin(), A_triplets.end() );
	//A.makeCompressed();

	cout << "forming b ...\n";
	for (int i = 0; i < curRow; i++) {
		b(i) = b_vec[i];
	}
	
	cout << "forming AT...\n";
	EigenSpMat AT = A.transpose();
	//AT.makeCompressed();
	
	cout << "multiplying ATA...\n";
	ATA = (AT * A).pruned();
	//ATA.makeCompressed();
	cout << "multiplying ATb...\n";
	ATb = AT * b;

	cout << "construct solver\n";
	ConjugateGradient<EigenSpMat> CG_sover(ATA);
	cout << "solving...\n";
	x = CG_sover.solve(ATb);
}

void LinearConstructor::output(Mat& HRimg) {
	cout << "output as Mat\n";
	HRimg = Mat::zeros(HR_rows, HR_cols, CV_64F);
	int curIdx = 0;
	for (int i = 0; i < HR_rows; i++) for (int j = 0; j < HR_cols; j++) {
		HRimg.at<double>(i, j) = x(curIdx);

		curIdx ++;
	}
}