#include "GPU.h"

void eigenSpMat2cudaCSR(EigenSpMat_Row& A, int *I, int *J, double *val, int N, int nz)
{
	// I = OuterStarts
	// J = InnerIndices
	// val = Values
	
	for (int i = 0; i < N+1; i++) {
		I[i] = A.outerIndexPtr()[i];
	}
	for (int i = 0; i < nz; i++) {
		J[i] = A.innerIndexPtr()[i];
	}
	for (int i = 0; i < nz; i++) {
		val[i] = A.valuePtr()[i];
	}
}

GPU_Context::GPU_Context()
{
	GPU_get = false;

	devID = findCudaDevice();
	if (devID < 0)
    {
        printf("exiting...\n");
		
		return;
        //exit(EXIT_SUCCESS);
    }

	cudaGetDeviceProperties(&deviceProp, devID);

	// Statistics about the GPU device
    printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
           deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);
	
    int version = (deviceProp.major * 0x10 + deviceProp.minor);

    if (version < 0x11)
    {
        printf("%s: requires a minimum CUDA compute 1.1 capability\n", "conjugateGradient");

        // cudaDeviceReset causes the driver to clean up all state. While
        // not mandatory in normal operation, it is good practice.  It is also
        // needed to ensure correct operation when the application is being
        // profiled. Calling cudaDeviceReset causes all profile data to be
        // flushed before the application exits
        cudaDeviceReset();
		return;
        //exit(EXIT_SUCCESS);
    }

	GPU_get = true;

	//////// 
	/* Get handle to the CUBLAS context */
	cublasHandle = 0;
    cublasStatus = cublasCreate(&cublasHandle);

	/* Get handle to the CUSPARSE context */
    cusparseHandle = 0;
    cusparseStatus = cusparseCreate(&cusparseHandle);

	// Sparse Mat Descr 
	descr = 0;
    cusparseStatus = cusparseCreateMatDescr(&descr);
	cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
}

GPU_Context::~GPU_Context()
{
	cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);
	// cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
	cudaDeviceReset();
	
}

void GPU_Context::ConjugateGradient_GPU (EigenSpMat& A_in, VectorXd& b_in, VectorXd& x_in)
{
	int A_M = 0, A_N = 0, A_nz = 0, *A_I = NULL, *A_J = NULL;
    double *A_val = NULL;
    const double tol = 1e-5f;
    const int max_iter = 10000;
    double *x_vec;
    double *b_vec;
    double a, b, na, r0, r1;
    int *d_A_col, *d_A_row;
    double *d_A_val, *d_x, dot, *d_b;
    double *d_p, *d_Cx, *d_r;
    int k;
    double alpha, beta, alpham1;

    /* transfer EigenSpMat to Cuda need */
	EigenSpMat_Row A_rowMajor = A_in;
	std::cout << "N: "<< A_rowMajor.rows() << std::endl;
	A_rowMajor.makeCompressed();
	A_M = A_rowMajor.rows();
	A_N = A_rowMajor.cols();
	A_nz = A_rowMajor.nonZeros();
    A_I = (int *)malloc(sizeof(int)*(A_M+1));
    A_J = (int *)malloc(sizeof(int)*A_nz);
    A_val = (double *)malloc(sizeof(double)*A_nz);
    
	eigenSpMat2cudaCSR(A_rowMajor, A_I, A_J, A_val, A_M, A_nz);

    x_vec = (double *)malloc(sizeof(double)*A_N);
    b_vec = (double *)malloc(sizeof(double)*A_M);

	for (int i = 0; i < A_M; i++) {
		b_vec[i] = b_in[i];
	}

    for (int i = 0; i < A_N; i++)
    {
        x_vec[i] = 0.0;
    }
	
	// copy A ------
	cudaMalloc((void **)&d_A_col, A_nz*sizeof(int));
	cudaMalloc((void **)&d_A_row, (A_M+1)*sizeof(int));
	cudaMalloc((void **)&d_A_val, A_nz*sizeof(double));

	cudaMemcpy(d_A_col, A_J, A_nz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_row, A_I, (A_M+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_val, A_val, A_nz*sizeof(double), cudaMemcpyHostToDevice);

	// C = ATA
	int baseC, nnzC; 
	//nnzTotalDevHostPtr points to host memory 
	int *nnzTotalDevHostPtr = &nnzC;
	int *csrRowPtrC, *csrColIndC;
	double *csrValC;
	cusparseSetPointerMode(cusparseHandle, CUSPARSE_POINTER_MODE_HOST);
	cudaMalloc((void**)&csrRowPtrC, sizeof(int)*(A_N+1));
	cusparseXcsrgemmNnz(cusparseHandle, 
		CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
		A_N, A_N, A_M,
		descr, A_nz, d_A_row, d_A_col,
		descr, A_nz, d_A_row, d_A_col,
		descr, csrRowPtrC, nnzTotalDevHostPtr );
	if (NULL != nnzTotalDevHostPtr){ 
		nnzC = *nnzTotalDevHostPtr;
	}
	else{ 
		cudaMemcpy(&nnzC, csrRowPtrC + A_N, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&baseC, csrRowPtrC, sizeof(int), cudaMemcpyDeviceToHost);
		nnzC -= baseC; 
	}
	cudaMalloc((void**)&csrColIndC, sizeof(int)*nnzC);
	cudaMalloc((void**)&csrValC, sizeof(double)*nnzC); 
	cusparseDcsrgemm(cusparseHandle, 
		CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
		A_N, A_N, A_M,
		descr, A_nz, d_A_val, d_A_row, d_A_col,
		descr, A_nz, d_A_val, d_A_row, d_A_col,
		descr, csrValC, csrRowPtrC, csrColIndC);

	//std::cout << "nnzC: " << nnzC << std::endl;

	// rhs = ATb
	alpha = 1.0;
	beta = 0;
	cudaMalloc((void **)&d_b, A_M*sizeof(double));
	cudaMemcpy(d_b, b_vec, A_M*sizeof(double), cudaMemcpyHostToDevice);
	cudaMalloc((void **)&d_r, A_N*sizeof(double));
	cusparseDcsrmv(cusparseHandle,
		CUSPARSE_OPERATION_TRANSPOSE,
		A_M, A_N, A_nz,
		&alpha, 
		descr, d_A_val, d_A_row, d_A_col,
		d_b, &beta, d_r);

	cublasStatus = cublasDdot(cublasHandle, A_N, d_r, 1, d_r, 1, &r1);
	//std::cout << "r1: " << r1 << std::endl;
	
	// release A in cuda (we need C = ATA now)
	cudaFree(d_A_col);
    cudaFree(d_A_row);
    cudaFree(d_A_val);

	//
	cudaMalloc((void **)&d_x, A_N*sizeof(double));
	cudaMalloc((void **)&d_p, A_N*sizeof(double));
	cudaMalloc((void **)&d_Cx, A_N*sizeof(double));

    cudaMemcpy(d_x, x_vec, A_N*sizeof(double), cudaMemcpyHostToDevice);

    alpha = 1.0;
    alpham1 = -1.0;
    beta = 0.0;
    r0 = 0.;

    cusparseDcsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, A_N, A_N, nnzC, &alpha, descr, csrValC, csrRowPtrC, csrColIndC, d_x, &beta, d_Cx);

    cublasDaxpy(cublasHandle, A_N, &alpham1, d_Cx, 1, d_r, 1);
    cublasStatus = cublasDdot(cublasHandle, A_N, d_r, 1, d_r, 1, &r1);

	//std::cout << "r1: " << r1 << std::endl;

    k = 1;

    while (r1 > tol*tol && k <= max_iter)
    {
        if (k > 1)
        {
            b = r1 / r0;
            cublasStatus = cublasDscal(cublasHandle, A_N, &b, d_p, 1);
            cublasStatus = cublasDaxpy(cublasHandle, A_N, &alpha, d_r, 1, d_p, 1);
        }
        else
        {
            cublasStatus = cublasDcopy(cublasHandle, A_N, d_r, 1, d_p, 1);
        }

        cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, A_N, A_N, nnzC, &alpha, descr, csrValC, csrRowPtrC, csrColIndC, d_p, &beta, d_Cx);
        cublasStatus = cublasDdot(cublasHandle, A_N, d_p, 1, d_Cx, 1, &dot);
        a = r1 / dot;

        cublasStatus = cublasDaxpy(cublasHandle, A_N, &a, d_p, 1, d_x, 1);
        na = -a;
        cublasStatus = cublasDaxpy(cublasHandle, A_N, &na, d_Cx, 1, d_r, 1);

        r0 = r1;
        cublasStatus = cublasDdot(cublasHandle, A_N, d_r, 1, d_r, 1, &r1);
        cudaThreadSynchronize();
        //printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
        k++;
    }

	printf("iteration = %3d, residual = %e\n", k, sqrt(r1));

    cudaMemcpy(x_vec, d_x, A_N*sizeof(double), cudaMemcpyDeviceToHost);

	// copy to xxx
	for (int i = 0; i < A_N; i++) {
		x_in(i) = x_vec[i];
	}

    free(A_I);
    free(A_J);
    free(A_val);
    free(x_vec);
    free(b_vec);
    
	cudaFree(csrColIndC);
    cudaFree(csrRowPtrC);
    cudaFree(csrValC);
	cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_Cx);

}

void GPU_Context::ConjugateGradient_GPU_squareMat (EigenSpMat& ATA, VectorXd& ATb, VectorXd& x_in)
{
	int M = 0, N = 0, nz = 0, *I = NULL, *J = NULL;
    double *val = NULL;
    const double tol = 1e-5f;
    const int max_iter = 10000;
    double *x;
    double *rhs;
    double a, b, na, r0, r1;
    int *d_col, *d_row;
    double *d_val, *d_x, dot;
    double *d_r, *d_p, *d_Ax;
    int k;
    double alpha, beta, alpham1;

    /* transfer EigenSpMat to Cuda need */
	EigenSpMat_Row ATA_rowMajor = ATA;
	std::cout << "N: "<< ATA_rowMajor.rows() << std::endl;
	ATA_rowMajor.makeCompressed();
	N = ATA_rowMajor.rows();
	nz = ATA_rowMajor.nonZeros();
    I = (int *)malloc(sizeof(int)*(N+1));
    J = (int *)malloc(sizeof(int)*nz);
    val = (double *)malloc(sizeof(double)*nz);
    
	eigenSpMat2cudaCSR(ATA_rowMajor, I, J, val, N, nz);

    x = (double *)malloc(sizeof(double)*N);
    rhs = (double *)malloc(sizeof(double)*N);

    for (int i = 0; i < N; i++)
    {
        rhs[i] = ATb[i];
        x[i] = 0.0;
    }
	
	cudaMalloc((void **)&d_col, nz*sizeof(int));
	cudaMalloc((void **)&d_row, (N+1)*sizeof(int));
	cudaMalloc((void **)&d_val, nz*sizeof(double));
	cudaMalloc((void **)&d_x, N*sizeof(double));
	cudaMalloc((void **)&d_r, N*sizeof(double));
	cudaMalloc((void **)&d_p, N*sizeof(double));
	cudaMalloc((void **)&d_Ax, N*sizeof(double));

    cudaMemcpy(d_col, J, nz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row, I, (N+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, val, nz*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, rhs, N*sizeof(double), cudaMemcpyHostToDevice);

    alpha = 1.0;
    alpham1 = -1.0;
    beta = 0.0;
    r0 = 0.;

    cusparseDcsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col, d_x, &beta, d_Ax);

    cublasDaxpy(cublasHandle, N, &alpham1, d_Ax, 1, d_r, 1);
    cublasStatus = cublasDdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);

    k = 1;

    while (r1 > tol*tol && k <= max_iter)
    {
        if (k > 1)
        {
            b = r1 / r0;
            cublasStatus = cublasDscal(cublasHandle, N, &b, d_p, 1);
            cublasStatus = cublasDaxpy(cublasHandle, N, &alpha, d_r, 1, d_p, 1);
        }
        else
        {
            cublasStatus = cublasDcopy(cublasHandle, N, d_r, 1, d_p, 1);
        }

        cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col, d_p, &beta, d_Ax);
        cublasStatus = cublasDdot(cublasHandle, N, d_p, 1, d_Ax, 1, &dot);
        a = r1 / dot;

        cublasStatus = cublasDaxpy(cublasHandle, N, &a, d_p, 1, d_x, 1);
        na = -a;
        cublasStatus = cublasDaxpy(cublasHandle, N, &na, d_Ax, 1, d_r, 1);

        r0 = r1;
        cublasStatus = cublasDdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
        cudaThreadSynchronize();
        //printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
        k++;
    }

	printf("iteration = %3d, residual = %e\n", k, sqrt(r1));

    cudaMemcpy(x, d_x, N*sizeof(double), cudaMemcpyDeviceToHost);


	// copy to xxx
	for (int i = 0; i < N; i++) {
		x_in(i) = x[i];
	}

    free(I);
    free(J);
    free(val);
    free(x);
    free(rhs);
    cudaFree(d_col);
    cudaFree(d_row);
    cudaFree(d_val);
    cudaFree(d_x);
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_Ax);

}

void L2GradientDescent_GPU (EigenSpMat& A, EigenSpMat& AT, EigenSpMat& CTC, VectorXd& b, VectorXd& xxx)
{
	int A_M = 0, A_N = 0, A_nz = 0, *A_I = NULL, *A_J = NULL;
	double* A_val;

	// This will pick the best possible CUDA capable device
	/*
	cudaDeviceProp deviceProp;
    int devID = findCudaDevice();

    if (devID < 0)
    {
        printf("exiting...\n");
		return;
        //exit(EXIT_SUCCESS);
    }

	cudaGetDeviceProperties(&deviceProp, devID);

    // Statistics about the GPU device
    printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
           deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);
	
    int version = (deviceProp.major * 0x10 + deviceProp.minor);

    if (version < 0x11)
    {
        printf("%s: requires a minimum CUDA compute 1.1 capability\n", "conjugateGradient");

        // cudaDeviceReset causes the driver to clean up all state. While
        // not mandatory in normal operation, it is good practice.  It is also
        // needed to ensure correct operation when the application is being
        // profiled. Calling cudaDeviceReset causes all profile data to be
        // flushed before the application exits
        cudaDeviceReset();
		return;
        //exit(EXIT_SUCCESS);
    }
	//*/

	EigenSpMat_Row A_Row = A;
	A_Row.makeCompressed();
	A_M = A_Row.cols();
	A_N = A_Row.rows();
	A_nz = A_Row.nonZeros();
    A_I = (int *)malloc(sizeof(int)*(A_N+1));
    A_J = (int *)malloc(sizeof(int)*A_nz);
    A_val = (double *)malloc(sizeof(double)*A_nz);
	eigenSpMat2cudaCSR(A_Row, A_I, A_J, A_val, A_N, A_nz);

	/*
	int A_M = 0, A_N = 0, A_nz = 0, *A_I = NULL, *A_J = NULL;
	double* A_val;

	EigenSpMat_Row AT_Row = AT;
	AT_Row.makeCompressed();
	A_M = AT_Row.cols();
	A_N = AT_Row.rows();
	A_nz = AT_Row.nonZeros();
    A_I = (int *)malloc(sizeof(int)*(A_N+1));
    A_J = (int *)malloc(sizeof(int)*A_nz);
    A_val = (double *)malloc(sizeof(double)*A_nz);
	eigenSpMat2cudaCSR(A_Row, A_I, A_J, A_val, A_N, A_nz);
	//*/

	int CTC_M = 0, CTC_N = 0, CTC_nz = 0, *CTC_I = NULL, *CTC_J = NULL;
	double* CTC_val;
	EigenSpMat_Row CTC_Row = CTC;
	CTC_Row.makeCompressed();
	CTC_N = CTC_Row.rows();
	CTC_M = CTC_N;
	CTC_nz = CTC_Row.nonZeros();
    CTC_I = (int *)malloc(sizeof(int)*(CTC_N+1));
    CTC_J = (int *)malloc(sizeof(int)*CTC_nz);
    CTC_val = (double *)malloc(sizeof(double)*CTC_nz);
	eigenSpMat2cudaCSR(CTC_Row, CTC_I, CTC_J, CTC_val, CTC_N, CTC_nz);

	double *x;
    double *b_vec;

	x = (double *)malloc(sizeof(double)*A_M);
    b_vec = (double *)malloc(sizeof(double)*A_N);

    for (int i = 0; i < A_N; i++)
    {
        b_vec[i] = b(i);
    }
	for (int i = 0; i < A_M; i++)
    {
        x[i] = 0.0;
    }


	/* Get handle to the CUBLAS context */
    cublasHandle_t cublasHandle = 0;
    cublasStatus_t cublasStatus;
    cublasStatus = cublasCreate(&cublasHandle);

    //checkCudaErrors(cublasStatus);

    /* Get handle to the CUSPARSE context */
    cusparseHandle_t cusparseHandle = 0;
    cusparseStatus_t cusparseStatus;
    cusparseStatus = cusparseCreate(&cusparseHandle);

    //checkCudaErrors(cusparseStatus);

    cusparseMatDescr_t descr = 0;
    cusparseStatus = cusparseCreateMatDescr(&descr);

    //checkCudaErrors(cusparseStatus);

    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

	int *d_A_col, *d_A_row, *d_CTC_col, *d_CTC_row;
    double *d_A_val, *d_CTC_val, *d_x, *d_b_vec;
	double *d_CTCx, *d_Ax, *d_AT_bAx, *d_step, *d_bAx;

	cudaMalloc((void **)&d_A_col, A_nz*sizeof(int));
	cudaMalloc((void **)&d_A_row, (A_N+1)*sizeof(int));
	cudaMalloc((void **)&d_A_val, A_nz*sizeof(double));

	cudaMalloc((void **)&d_CTC_col, CTC_nz*sizeof(int));
	cudaMalloc((void **)&d_CTC_row, (CTC_N+1)*sizeof(int));
	cudaMalloc((void **)&d_CTC_val, CTC_nz*sizeof(double));

	cudaMalloc((void **)&d_x, A_M*sizeof(double));
	cudaMalloc((void **)&d_CTCx, A_M*sizeof(double));
	cudaMalloc((void **)&d_AT_bAx, A_M*sizeof(double));
	cudaMalloc((void **)&d_step, A_M*sizeof(double));

	cudaMalloc((void **)&d_b_vec, A_N*sizeof(double));
	cudaMalloc((void **)&d_Ax, A_N*sizeof(double));
	cudaMalloc((void **)&d_bAx, A_N*sizeof(double));

    cudaMemcpy(d_A_col, A_J, A_nz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_row, A_I, (A_N+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_val, A_val, A_nz*sizeof(double), cudaMemcpyHostToDevice);

	cudaMemcpy(d_CTC_col, CTC_J, CTC_nz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_CTC_row, CTC_I, (CTC_N+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_CTC_val, CTC_val, CTC_nz*sizeof(double), cudaMemcpyHostToDevice);

	cudaMemcpy(d_x, x, A_M*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b_vec, b_vec, A_N*sizeof(double), cudaMemcpyHostToDevice);


	int k = 1;
	int max_iter = 10000;
	double tol = 1e-4;
	double err = 1e8;
    double alpha, beta;

    while (err > tol && k <= max_iter)
    {
        // x_n1 = x_n + 0.5 * ( AT * (b - (A * x_n)) - (CTC * x_n) ); // 26 sec
		// A * x_n
		alpha = 1;
		beta = 0;
		cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
			A_N, A_M, A_nz, &alpha, descr, d_A_val, d_A_row, d_A_col, d_x, &beta, d_Ax);

		cudaThreadSynchronize();

		// (b - (A * x_n))
		cublasStatus = cublasDcopy(cublasHandle, A_N, d_b_vec, 1, d_bAx, 1);
		cudaThreadSynchronize();
		alpha = -1;
		cublasDaxpy(cublasHandle, A_N, &alpha, d_Ax, 1, d_bAx, 1);
		cudaThreadSynchronize();
		// CTC * x_n
		alpha = 1;
		beta = 0;
		cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
			CTC_N, CTC_M, CTC_nz, &alpha, descr, d_CTC_val, d_CTC_row, d_CTC_col, d_x, &beta, d_CTCx);
		cudaThreadSynchronize();
		// AT * (b - (A * x_n)) - CTC * x_n
		alpha = 1;
		beta = -1;
		cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE,
			A_N, A_M, A_nz, &alpha, descr, d_A_val, d_A_row, d_A_col, d_bAx, &beta, d_CTCx);
		cudaThreadSynchronize();
		// now, diff = stores in CTCx
		alpha = 0.5;
		cublasDaxpy(cublasHandle, A_M, &alpha, d_CTCx, 1, d_x, 1);
		// calc err^2
		cublasDnrm2(cublasHandle, A_M, d_CTCx, 1, &err);
		err = 0.5 * sqrt(err) / A_M;

        cudaThreadSynchronize();

        //printf("iteration = %3d, residual = %e\n", k, err);
        k++;
    }
	printf("iteration = %3d, residual = %e\n", k, err);

	cudaMemcpy(x, d_x, A_M*sizeof(double), cudaMemcpyDeviceToHost);

	// copy to xxx
	for (int i = 0; i < A_M; i++) {
		xxx(i) = x[i];
	}

	//cusparseDestroy(cusparseHandle);
    //cublasDestroy(cublasHandle);

    free(A_I);
    free(A_J);
    free(A_val);
	free(CTC_I);
    free(CTC_J);
    free(CTC_val);
    free(x);
	free(b_vec);

    cudaFree(d_A_col);
    cudaFree(d_A_row);
    cudaFree(d_A_val);
	cudaFree(d_CTC_col);
    cudaFree(d_CTC_row);
    cudaFree(d_CTC_val);
    cudaFree(d_x);
    cudaFree(d_b_vec);
	cudaFree(d_CTCx);
    cudaFree(d_Ax);
    cudaFree(d_AT_bAx);
	cudaFree(d_step);
    cudaFree(d_bAx);

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    //cudaDeviceReset();
}void GPU_Context::L2GradientDescent_GPU (EigenSpMat& A, EigenSpMat& CTC, VectorXd& b, VectorXd& xxx)
{
	int A_M = 0, A_N = 0, A_nz = 0, *A_I = NULL, *A_J = NULL;
	double* A_val;

	EigenSpMat_Row A_Row = A;
	A_Row.makeCompressed();
	A_M = A_Row.cols();
	A_N = A_Row.rows();
	A_nz = A_Row.nonZeros();
    A_I = (int *)malloc(sizeof(int)*(A_N+1));
    A_J = (int *)malloc(sizeof(int)*A_nz);
    A_val = (double *)malloc(sizeof(double)*A_nz);
	eigenSpMat2cudaCSR(A_Row, A_I, A_J, A_val, A_N, A_nz);

	int CTC_M = 0, CTC_N = 0, CTC_nz = 0, *CTC_I = NULL, *CTC_J = NULL;
	double* CTC_val;
	EigenSpMat_Row CTC_Row = CTC;
	CTC_Row.makeCompressed();
	CTC_N = CTC_Row.rows();
	CTC_M = CTC_N;
	CTC_nz = CTC_Row.nonZeros();
    CTC_I = (int *)malloc(sizeof(int)*(CTC_N+1));
    CTC_J = (int *)malloc(sizeof(int)*CTC_nz);
    CTC_val = (double *)malloc(sizeof(double)*CTC_nz);
	eigenSpMat2cudaCSR(CTC_Row, CTC_I, CTC_J, CTC_val, CTC_N, CTC_nz);

	double *x;
    double *b_vec;

	x = (double *)malloc(sizeof(double)*A_M);
    b_vec = (double *)malloc(sizeof(double)*A_N);

    for (int i = 0; i < A_N; i++)
    {
        b_vec[i] = b(i);
    }
	for (int i = 0; i < A_M; i++)
    {
        x[i] = 0.0;
    }

	int *d_A_col, *d_A_row, *d_CTC_col, *d_CTC_row;
    double *d_A_val, *d_CTC_val, *d_x, *d_b_vec;
	double *d_CTCx, *d_Ax, *d_AT_bAx, *d_step, *d_bAx;

	cudaMalloc((void **)&d_A_col, A_nz*sizeof(int));
	cudaMalloc((void **)&d_A_row, (A_N+1)*sizeof(int));
	cudaMalloc((void **)&d_A_val, A_nz*sizeof(double));

	cudaMalloc((void **)&d_CTC_col, CTC_nz*sizeof(int));
	cudaMalloc((void **)&d_CTC_row, (CTC_N+1)*sizeof(int));
	cudaMalloc((void **)&d_CTC_val, CTC_nz*sizeof(double));

	cudaMalloc((void **)&d_x, A_M*sizeof(double));
	cudaMalloc((void **)&d_CTCx, A_M*sizeof(double));
	cudaMalloc((void **)&d_AT_bAx, A_M*sizeof(double));
	cudaMalloc((void **)&d_step, A_M*sizeof(double));

	cudaMalloc((void **)&d_b_vec, A_N*sizeof(double));
	cudaMalloc((void **)&d_Ax, A_N*sizeof(double));
	cudaMalloc((void **)&d_bAx, A_N*sizeof(double));

    cudaMemcpy(d_A_col, A_J, A_nz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_row, A_I, (A_N+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_val, A_val, A_nz*sizeof(double), cudaMemcpyHostToDevice);

	cudaMemcpy(d_CTC_col, CTC_J, CTC_nz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_CTC_row, CTC_I, (CTC_N+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_CTC_val, CTC_val, CTC_nz*sizeof(double), cudaMemcpyHostToDevice);

	cudaMemcpy(d_x, x, A_M*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b_vec, b_vec, A_N*sizeof(double), cudaMemcpyHostToDevice);


	int k = 1;
	int max_iter = 10000;
	double tol = 1e-4;
	double err = 1e8;
    double alpha, beta;

    while (err > tol && k <= max_iter)
    {
        // x_n1 = x_n + 0.5 * ( AT * (b - (A * x_n)) - (CTC * x_n) ); // 26 sec
		// A * x_n
		alpha = 1;
		beta = 0;
		cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
			A_N, A_M, A_nz, &alpha, descr, d_A_val, d_A_row, d_A_col, d_x, &beta, d_Ax);

		cudaThreadSynchronize();

		// (b - (A * x_n))
		cublasStatus = cublasDcopy(cublasHandle, A_N, d_b_vec, 1, d_bAx, 1);
		cudaThreadSynchronize();
		alpha = -1;
		cublasDaxpy(cublasHandle, A_N, &alpha, d_Ax, 1, d_bAx, 1);
		cudaThreadSynchronize();
		// CTC * x_n
		alpha = 1;
		beta = 0;
		cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
			CTC_N, CTC_M, CTC_nz, &alpha, descr, d_CTC_val, d_CTC_row, d_CTC_col, d_x, &beta, d_CTCx);
		cudaThreadSynchronize();
		// AT * (b - (A * x_n)) - CTC * x_n
		alpha = 1;
		beta = -1;
		cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE,
			A_N, A_M, A_nz, &alpha, descr, d_A_val, d_A_row, d_A_col, d_bAx, &beta, d_CTCx);
		cudaThreadSynchronize();
		// now, diff = stores in CTCx
		alpha = 0.5;
		cublasDaxpy(cublasHandle, A_M, &alpha, d_CTCx, 1, d_x, 1);
		// calc err^2
		cublasDnrm2(cublasHandle, A_M, d_CTCx, 1, &err);
		err = 0.5 * sqrt(err) / A_M;

        cudaThreadSynchronize();

        //printf("iteration = %3d, residual = %e\n", k, err);
        k++;
    }
	printf("iteration = %3d, residual = %e\n", k, err);

	cudaMemcpy(x, d_x, A_M*sizeof(double), cudaMemcpyDeviceToHost);

	// copy to xxx
	for (int i = 0; i < A_M; i++) {
		xxx(i) = x[i];
	}

    free(A_I);
    free(A_J);
    free(A_val);
	free(CTC_I);
    free(CTC_J);
    free(CTC_val);
    free(x);
	free(b_vec);

    cudaFree(d_A_col);
    cudaFree(d_A_row);
    cudaFree(d_A_val);
	cudaFree(d_CTC_col);
    cudaFree(d_CTC_row);
    cudaFree(d_CTC_val);
    cudaFree(d_x);
    cudaFree(d_b_vec);
	cudaFree(d_CTCx);
    cudaFree(d_Ax);
    cudaFree(d_AT_bAx);
	cudaFree(d_step);
    cudaFree(d_bAx);
}