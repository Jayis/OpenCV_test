#pragma once

#include <Eigen\SparseCore>
#include <Eigen\Sparse>

#define EX_small (1e-8)
#define EX_big (1e8)
#define SQR(x) ((x)*(x))

typedef Eigen::Triplet<double> T;
//typedef Eigen::SparseMatrix<double, Eigen::RowMajor> EigenSpRowMat;
//typedef Eigen::SparseMatrix<double, Eigen::ColMajor> EigenSpColMat;
typedef Eigen::SparseMatrix<double> EigenSpMat;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> EigenSpMat_Row;