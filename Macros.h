#pragma once

#include <Eigen\SparseCore>
#include <Eigen\Sparse>

#define EXsmall (1e-8)
#define SQR(x) ((x)*(x))

typedef Eigen::Triplet<double> T;
//typedef Eigen::SparseMatrix<double, Eigen::RowMajor> EigenSpRowMat;
//typedef Eigen::SparseMatrix<double, Eigen::ColMajor> EigenSpColMat;
typedef Eigen::SparseMatrix<double> EigenSpMat;
