#pragma once

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
//#include <math.h>

/* Using updated (v2) interfaces to cublas */
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

// Utilities and system includes
//#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
//#include <helper_cuda.h>       // helper function CUDA error checking and initialization

#include "Macros.h"
#include "My_CudaHelper.h"

using namespace Eigen;

void eigenSpMat2cudaCSR(EigenSpMat_Row& A, int *I, int *J, double *val, int N, int nz);

class GPU_Context
{
public:
	GPU_Context();
	~GPU_Context();

	void ConjugateGradient_GPU (EigenSpMat& A, VectorXd& b_in, VectorXd& x_in);
	void ConjugateGradient_GPU_squareMat (EigenSpMat& ATA, VectorXd& ATb, VectorXd& x_in);
	void L2GradientDescent_GPU (EigenSpMat& A, EigenSpMat& CTC, VectorXd& b, VectorXd& xxx);

private:
	bool GPU_get;
	//
	int devID;
	cudaDeviceProp deviceProp;
	// cublas
	cublasHandle_t cublasHandle;
	cublasStatus_t cublasStatus;
	// cusparse
	cusparseHandle_t cusparseHandle;
    cusparseStatus_t cusparseStatus;
	// mat descr
	cusparseMatDescr_t descr;
};