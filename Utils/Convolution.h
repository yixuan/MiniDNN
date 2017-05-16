#ifndef UTILS_CONVOLUTION_H_
#define UTILS_CONVOLUTION_H_

#include <Eigen/Core>
#include <vector>
#include "../Config.h"

// A simple wrapper of Eigen::Map
// We need this since Eigen::Map does not have the assignment operator we need
// - Eigen::Map<const Eigen::Matrix> does not have an assignment operator
// - Eigen::Map<Eigen::Matrix> does have an assignment operator, but it means copying the data of rhs to lhs
// As a result, we cannot add an Eigen::Map object to std::vector using push_back()
class MatWrapper
{
private:
    Scalar* m_data;
    int     m_rows;
    int     m_cols;
public:
    typedef Scalar* PointerType;
    typedef Eigen::Map< Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> > MapMatType;
    MatWrapper(Scalar* data, int rows, int cols) : m_data(data), m_rows(rows), m_cols(cols) {}
    int rows() const { return m_rows; }
    int cols() const { return m_cols; }
    MapMatType get() { return MapMatType(m_data, m_rows, m_cols); }
};

class ConstMatWrapper
{
private:
    const Scalar* m_data;
    int           m_rows;
    int           m_cols;
public:
    typedef const Scalar* PointerType;
    typedef Eigen::Map< const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> > MapMatType;
    ConstMatWrapper(const Scalar* data, int rows, int cols) : m_data(data), m_rows(rows), m_cols(cols) {}
    int rows() const { return m_rows; }
    int cols() const { return m_cols; }
    MapMatType get() const { return MapMatType(m_data, m_rows, m_cols); }
};

// Convert a 1D vector into a 4D tensor
// The first two subscripts are implemented using std::vector, and the next two
// dimensions use mapped matrices. Memory layout is as follows:
// t[0, 0, , ,] => t[0, 1, , , ] => ... => t[0, d2 - 1, , , ] => t[1, 0, , , ] => t[1, 1, , , ] => ...
//
// MapMatType is either a writable mapped mat (Map<Matrix>) or a read-only one (Map<const Matrix>)
template <typename MatWrapperType>
void vector_to_tensor_4d(
    typename MatWrapperType::PointerType src, const int d1, const int d2, const int d3, const int d4,
    std::vector< std::vector<MatWrapperType> >& tensor
)
{
    const int inner_mat_size = d3 * d4;

    tensor.clear();
    tensor.reserve(d1);
    for(int i = 0; i < d1; i++)
    {
        tensor.push_back(std::vector<MatWrapperType>());
        tensor[i].reserve(d2);
        for(int j = 0; j < d2; j++)
        {
            typename MatWrapperType::PointerType slice_start = src + (i * d2 + j) * inner_mat_size;
            tensor[i].push_back(MatWrapperType(slice_start, d3, d4));
        }
    }
}

// Convolution using the "valid" rule
// "dest" should be properly sized and zeroed
// dest.rows() == src.rows() - Krows + 1
// dest.cols() == src.cols() - Kcols + 1
template <int Krows, int Kcols>
void convolve_valid(
    const ConstMatWrapper& src,
    const Eigen::Matrix<Scalar, Krows, Kcols>& kernel,
    MatWrapper& dest)
{
    typedef Eigen::Map< const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> > ConstMapMat;
    typedef Eigen::Map< Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> > MapMat;

	const int dest_rows = dest.rows();
    const int dest_cols = dest.cols();

    ConstMapMat msrc = src.get();
    MapMat mdest = dest.get();
	for(int j = 0; j < dest_cols; j++)
	{
		for(int i = 0; i < dest_rows; i++)
		{
			mdest.coeffRef(i, j) += msrc.block<Krows, Kcols>(i, j).cwiseProduct(kernel).sum();
		}
	}
}
// Overload for mapped matrix
// KernelType is either MatWrapper or ConstMatWrapper
template <typename KernelType>
void convolve_valid(
    const ConstMatWrapper& src,
    KernelType& kernel,
    MatWrapper& dest)
{
    typedef Eigen::Map< const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> > ConstMapMat;
    typedef Eigen::Map< Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> > MapMat;

	const int dest_rows = dest.rows();
    const int dest_cols = dest.cols();
    const int K_rows = kernel.rows();
    const int K_cols = kernel.cols();

    ConstMapMat msrc = src.get();
    typename KernelType::MapMatType mkernel = kernel.get();
    MapMat mdest = dest.get();
	for(int j = 0; j < dest_cols; j++)
	{
		for(int i = 0; i < dest_rows; i++)
		{
			mdest.coeffRef(i, j) += msrc.block(i, j, K_rows, K_cols).cwiseProduct(mkernel).sum();
		}
	}
}

// Convolution using the "full" rule, with kernel rotated
// "dest" should be properly sized and zeroed
// dest.rows() == src.rows() + Krows - 1
// dest.cols() == src.cols() + Kcols - 1
template <int Krows, int Kcols>
void convolve_full(
    const ConstMatWrapper& src,
    const Eigen::Matrix<Scalar, Krows, Kcols>& kernel,
    MatWrapper& dest)
{
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Map<Matrix> MapMat;

    const int dest_rows = dest.rows();
    const int dest_cols = dest.cols();
    Matrix src_pad = Matrix::Zero(dest_rows + Krows - 1, dest_cols + Kcols - 1);
    src_pad.block(Krows - 1, Kcols - 1, src.rows(), src.cols()).noalias() = src.get();
    MapMat mdest = dest.get();

    for(int j = 0; j < dest_cols; j++)
	{
		for(int i = 0; i < dest_rows; i++)
		{
			mdest.coeffRef(i, j) += src_pad.block<Krows, Kcols>(i, j).cwiseProduct(kernel.reverse()).sum();
		}
	}
}
// Overload for mapped matrix
template <typename KernelType>
void convolve_full(
    const ConstMatWrapper& src,
    KernelType& kernel,
    MatWrapper& dest)
{
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Map<Matrix> MapMat;

    const int K_rows = kernel.rows();
    const int K_cols = kernel.cols();
    const int dest_rows = dest.rows();
    const int dest_cols = dest.cols();
    Matrix src_pad = Matrix::Zero(dest_rows + K_rows - 1, dest_cols + K_cols - 1);
    src_pad.block(K_rows - 1, K_cols - 1, src.rows(), src.cols()).noalias() = src.get();
    MapMat mdest = dest.get();
    typename KernelType::MapMatType mkernel = kernel.get();

    for(int j = 0; j < dest_cols; j++)
	{
		for(int i = 0; i < dest_rows; i++)
		{
			mdest.coeffRef(i, j) += src_pad.block(i, j, K_rows, K_cols).cwiseProduct(mkernel.reverse()).sum();
		}
	}
}


#endif /* UTILS_CONVOLUTION_H_ */
