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

using namespace Eigen;

void eigenSpMat2cudaCSR(EigenSpMat_Row& ATA, int *I, int *J, double *val, int N, int nz);

void ConjugateGradient_GPU (EigenSpMat& ATA, VectorXd& ATb, VectorXd& xxx);

void L2GradientDescent_GPU (EigenSpMat& A, EigenSpMat& AT, EigenSpMat& CTC, VectorXd& b, VectorXd& xxx);