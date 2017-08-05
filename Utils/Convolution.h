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

// We assume the following memory layout:
// There are 'n_obs' images, each with 'in_channels' channels
// Each channel has 'channel_rows' rows and 'channel_cols' columns
// The data starts from 'src'
// If 'image_outer_loop == true', the data first iterates on channels,
// and then images:
/*
 * ###############################################################
 * #           #           #           #           #
 * # channel 1 # channel 2 # channel 3 # channel 1 # ...
 * #           #           #           #           #
 * ###############################################################
 * |<------------ image 1 ------------>|<------------ image 2 ----
 */
// If 'image_outer_loop == false', the layout looks like below:
/*
 * ###############################################################
 * #           #           #           #           #
 * #  image 1  #  image 2  #  image 3  #  image 1  # ...
 * # channel 1 # channel 1 # channel 1 # channel 2 #
 * #           #           #           #           #
 * ###############################################################
 * |<----------- channel 1 ----------->|<----------- channel 2 ----
 */
//
// Then we assume there are 'out_channels' output channels, so in total
// we have 'in_channels * out_channels' filters for each image
// Each filter has 'filter_rows' rows and 'filter_cols' columns
// Filters start from 'filter_data'. The layout looks like below, with each
// block consisting of 'filter_rows * filter_cols' elements
/*
 * #########################################################################
 * #               #               #               #               #
 * # out channel 1 # out channel 2 # out channel 3 # out channel 1 # ...
 * #               #               #               #               #
 * #########################################################################
 * |<---------------- in channel 1 --------------->|<---------------- in channel 2 ----
 */
//
// Convolution results from different input channels are summed up to produce the
// result for each output channel
// Convolution results for different output channels are concatenated to preoduce
// the result for each image
//
// The final result is written to the memory pointed by 'dest', with a similar
// layout to 'src' in the 'image_outer_loop == true' case
//
// Memory efficient convolution (MEC)
// Algorithm is based on https://arxiv.org/abs/1706.06873
//
// First define a simple structure to store the various dimensions of convolution
struct ConvDims
{
    // Input parameters
    const int in_channels;
    const int out_channels;
    const int channel_rows;
    const int channel_cols;
    const int filter_rows;
    const int filter_cols;
    // Image dimension -- one observation with all channels
	const int img_rows;
	const int img_cols;
	// Dimension of the convolution result for each output channel
	const int conv_rows;
	const int conv_cols;

    ConvDims(
        const int in_channels_, const int out_channels_,
        const int channel_rows_, const int channel_cols_,
        const int filter_rows_, const int filter_cols_
    ) :
        in_channels(in_channels_), out_channels(out_channels_),
        channel_rows(channel_rows_), channel_cols(channel_cols_),
        filter_rows(filter_rows_), filter_cols(filter_cols_),
        img_rows(channel_rows_), img_cols(in_channels_ * channel_cols_),
        conv_rows(channel_rows_ - filter_rows_ + 1), conv_cols(channel_cols_ - filter_cols_ + 1)
    {}
};
// Then define a helper function to "flatten" source images as described in the paper
// 'flat_mat' will be resized and overwritten
inline void flatten_mat(
    const ConvDims& dim, const Scalar* src, const bool image_outer_loop, const int n_obs,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& flat_mat
)
{
    // Transform original matrix to "lower" form as described in the MEC paper
	// I feel that it is better called the "flat" form
	const int flat_rows = dim.conv_rows * n_obs;
	const int flat_cols = dim.filter_rows * dim.img_cols;
    flat_mat.resize(flat_rows, flat_cols);
	Scalar* flat_mat_data = flat_mat.data();

    const int img_obs_cols = n_obs * dim.img_cols;
    const int flat_size_per_obs = dim.conv_rows * flat_cols;
    // Number of bytes in the segment that will be copied at one time
    const std::size_t copy_bytes = sizeof(Scalar) * dim.filter_rows;

    if(image_outer_loop)
    {
        const Scalar* reader_col_start = src;
        for(int i = 0; i < img_obs_cols; i++, reader_col_start += dim.img_rows)
        {
            const Scalar* reader = reader_col_start;
            const Scalar* const reader_end = reader + dim.conv_rows;
            const int obs = i / dim.img_cols;
            const int writer_col = (i % dim.img_cols) * dim.filter_rows;
            Scalar* writer = flat_mat_data + flat_size_per_obs * obs + writer_col;
            for(; reader < reader_end; reader++, writer += flat_cols)
                std::memcpy(writer, reader, copy_bytes);
        }
    } else {
        const int channel_obs_cols = n_obs * dim.channel_cols;
        const int flat_cols_per_obs_channel = dim.filter_rows * dim.channel_cols;
        const Scalar* reader_col_start = src;
        for(int i = 0; i < img_obs_cols; i++, reader_col_start += dim.img_rows)
        {
            const Scalar* reader = reader_col_start;
            const Scalar* const reader_end = reader + dim.conv_rows;
            const int obs = (i % channel_obs_cols) / dim.channel_cols;
            const int channel = i / channel_obs_cols;
            const int writer_col = (i % dim.channel_cols) * dim.filter_rows;
            Scalar* writer = flat_mat_data + flat_size_per_obs * obs + flat_cols_per_obs_channel * channel + writer_col;
            for(; reader < reader_end; reader++, writer += flat_cols)
                std::memcpy(writer, reader, copy_bytes);
        }
    }
}
// The main convolution function
inline void convolve_valid(
    const ConvDims& dim,
	const Scalar* src, const bool image_outer_loop, const int n_obs, const Scalar* filter_data,
	Scalar* dest)
{
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RMatrix;
	typedef Eigen::Map<const Matrix> ConstMapMat;

    // Flat matrix
    RMatrix flat_mat;
    flatten_mat(dim, src, image_outer_loop, n_obs, flat_mat);

	// Filters
	const int filter_size = dim.filter_rows * dim.filter_cols;
	ConstMapMat filter(filter_data, filter_size, dim.in_channels * dim.out_channels);

	// Compute the convolution result
    const int res_rows = flat_mat.rows();
    const int res_cols = dim.conv_cols * dim.out_channels;
	Matrix res = Matrix::Zero(res_rows, res_cols);
	int filter_start_col = 0;
	int flat_mat_channel_start_col = 0;
	const int flat_mat_channel_step = dim.filter_rows * dim.channel_cols;
	for(int i = 0; i < dim.in_channels; i++, flat_mat_channel_start_col += flat_mat_channel_step, filter_start_col += dim.out_channels)
	{
		int flat_mat_start_col = flat_mat_channel_start_col;
		int res_start_col = 0;
		for(int j = 0; j < dim.conv_cols; j++, flat_mat_start_col += dim.filter_rows, res_start_col += dim.out_channels)
		{
			res.block(0, res_start_col, res_rows, dim.out_channels).noalias() +=
                flat_mat.block(0, flat_mat_start_col, res_rows, filter_size) *
                filter.block(0, filter_start_col, filter_size, dim.out_channels);
		}
	}

	// The layout of 'res' is very complicated
	/*
	 * obs0_out0[0, 0] obs0_out1[0, 0] obs0_out2[0, 0] obs0_out0[0, 1] obs0_out1[0, 1] obs0_out2[0, 1] ...
	 * obs0_out0[1, 0] obs0_out1[1, 0] obs0_out2[1, 0] obs0_out0[1, 1] obs0_out1[1, 1] obs0_out2[1, 1] ...
	 * obs0_out0[2, 0] obs0_out1[2, 0] obs0_out2[2, 0] obs0_out0[2, 1] obs0_out1[2, 1] obs0_out2[2, 1] ...
	 * obs1_out0[0, 0] obs1_out1[0, 0] obs1_out2[0, 0] obs1_out0[0, 1] obs1_out1[0, 1] obs1_out2[0, 1] ...
	 * obs1_out0[1, 0] obs1_out1[1, 0] obs1_out2[1, 0] obs1_out0[1, 1] obs1_out1[1, 1] obs1_out2[1, 1] ...
	 * obs1_out0[2, 0] obs1_out1[2, 0] obs1_out2[2, 0] obs1_out0[2, 1] obs1_out1[2, 1] obs1_out2[2, 1] ...
	 * ...
     *
	 */
	// obs<k>_out<l> means the convolution result of the k-th image on the l-th output channel
	// [i, j] gives the matrix indices

	// The destination has the layout
	/*
	 * obs0_out0[0, 0] obs0_out0[0, 1] obs0_out0[0, 2] obs0_out1[0, 0] obs0_out1[0, 1] obs0_out1[0, 2] ...
	 * obs0_out0[1, 0] obs0_out0[1, 1] obs0_out0[1, 2] obs0_out1[1, 0] obs0_out1[1, 1] obs0_out1[1, 2] ...
	 * obs0_out0[2, 0] obs0_out0[2, 1] obs0_out0[2, 2] obs0_out1[2, 0] obs0_out1[2, 1] obs0_out1[2, 2] ...
	 *
	 */
    // which in a larger scale looks like
    // [obs0_out0 obs0_out1 obs0_out2 obs1_out0 obs1_out1 obs1_out2 obs2_out0 ...]

	// Copy data to destination
    // dest[a, b] corresponds to obs<k>_out<l>[i, j]
    // where k = b / (conv_cols * out_channels),
    //       l = (b % (conv_cols * out_channels)) / conv_cols
    //       i = a,
    //       j = b % conv_cols
    // and then obs<k>_out<l>[i, j] corresponds to res[c, d]
    // where c = k * conv_rows + i,
    //       d = j * out_channels + l
    const int dest_rows = dim.conv_rows;
    const int dest_cols = res_cols * n_obs;
    const Scalar* res_data = res.data();
    const std::size_t copy_bytes = sizeof(Scalar) * dest_rows;
    for(int b = 0; b < dest_cols; b++, dest += dest_rows)
    {
        const int k = b / res_cols;
        const int l = (b % res_cols) / dim.conv_cols;
        const int j = b % dim.conv_cols;
        const int d = j * dim.out_channels + l;
        const int res_col_head = d * res_rows;
        std::memcpy(dest, res_data + res_col_head + k * dim.conv_rows, copy_bytes);
    }
}


#endif /* UTILS_CONVOLUTION_H_ */
