#pragma once

#include <Eigen\Sparse>

#define EXsmall (1e-8)
#define SQR(x) ((x)*(x))

typedef Eigen::Triplet<double> T;
typedef Eigen::SparseMatrix<double> EigenSpMat;