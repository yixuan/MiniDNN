#ifndef UTILS_CONVOLUTION_H_
#define UTILS_CONVOLUTION_H_

#include <Eigen/Core>
#include <vector>
#include "../Config.h"

// Convert a 1D vector into a 4D tensor
// The first two subscripts are implemented using std::vector, and the next two
// dimensions use mapped matrices. Memory layout is as follows:
// t[0, 0, , ,] => t[0, 1, , , ] => ... => t[0, d2 - 1, , , ] => t[1, 0, , , ] => t[1, 1, , , ] => ...
//
// MapMatType is either a writable mapped mat (Map<Matrix>) or a read-only one (Map<const Matrix>)
template <typename MapMatType>
void vector_to_tensor_4d(
    typename MapMatType::PointerType src, const int d1, const int d2, const int d3, const int d4,
    std::vector< std::vector<MapMatType> >& tensor
)
{
    const int inner_mat_size = d3 * d4;

    tensor.clear();
    tensor.reserve(d1);
    for(int i = 0; i < d1; i++)
    {
        tensor.push_back(std::vector<MapMatType>());
        tensor[i].reserve(d2);
        for(int j = 0; j < d2; j++)
        {
            typename MapMatType::PointerType slice_start = src + (i * d2 + j) * inner_mat_size;
            tensor[i].push_back(MapMatType(slice_start, d3, d4));
        }
    }
}

// Convolution using the "valid" rule
// "dest" should be properly sized and zeroed
// dest.rows() == src.rows() - Krows + 1
// dest.cols() == src.cols() - Kcols + 1
template <int Krows, int Kcols>
void convolve_valid(
    const Eigen::Map< const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> >& src,
    const Eigen::Matrix<Scalar, Krows, Kcols>& kernel,
    Eigen::Map< Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> >& dest)
{
	const int dest_rows = dest.rows();
    const int dest_cols = dest.cols();

	for(int j = 0; j < dest_cols; j++)
	{
		for(int i = 0; i < dest_rows; i++)
		{
			dest.coeffRef(i, j) += src.block<Krows, Kcols>(i, j).cwiseProduct(kernel).sum();
		}
	}
}
// Specialization for dynamic kernel matrix
template <>
void convolve_valid<Eigen::Dynamic, Eigen::Dynamic>(
    const Eigen::Map< const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> >& src,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& kernel,
    Eigen::Map< Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> >& dest)
{
	const int dest_rows = dest.rows();
    const int dest_cols = dest.cols();
    const int K_rows = kernel.rows();
    const int K_cols = kernel.cols();

	for(int j = 0; j < dest_cols; j++)
	{
		for(int i = 0; i < dest_rows; i++)
		{
			dest.coeffRef(i, j) += src.block(i, j, K_rows, K_cols).cwiseProduct(kernel).sum();
		}
	}
}

// Convolution using the "full" rule
// "dest" should be properly sized and zeroed
// dest.rows() == src.rows() + Krows - 1
// dest.cols() == src.cols() + Kcols - 1
template <int Krows, int Kcols>
void convolve_full(
    const Eigen::Map< const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> >& src,
    const Eigen::Matrix<Scalar, Krows, Kcols>& kernel,
    Eigen::Map< Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> >& dest)
{
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

    const int dest_rows = dest.rows();
    const int dest_cols = dest.cols();
    Matrix src_pad = Matrix::Zero(dest_rows + Krows - 1, dest_cols + Kcols - 1);
    src_pad.block(Krows - 1, Kcols - 1, src.rows(), src.cols()).noalias() = src;

    for(int j = 0; j < dest_cols; j++)
	{
		for(int i = 0; i < dest_rows; i++)
		{
			dest.coeffRef(i, j) += src_pad.block<Krows, Kcols>(i, j).cwiseProduct(kernel).sum();
		}
	}
}
// Specialization for dynamic kernel matrix
template <>
void convolve_full<Eigen::Dynamic, Eigen::Dynamic>(
    const Eigen::Map< const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> >& src,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& kernel,
    Eigen::Map< Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> >& dest)
{
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

    const int K_rows = kernel.rows();
    const int K_cols = kernel.cols();
    const int dest_rows = dest.rows();
    const int dest_cols = dest.cols();
    Matrix src_pad = Matrix::Zero(dest_rows + K_rows - 1, dest_cols + K_cols - 1);
    src_pad.block(K_rows - 1, K_cols - 1, src.rows(), src.cols()).noalias() = src;

    for(int j = 0; j < dest_cols; j++)
	{
		for(int i = 0; i < dest_rows; i++)
		{
			dest.coeffRef(i, j) += src_pad.block(i, j, K_rows, K_cols).cwiseProduct(kernel).sum();
		}
	}
}


#endif /* UTILS_CONVOLUTION_H_ */
