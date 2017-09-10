#ifndef LAYER_MAXPOOLING_H_
#define LAYER_MAXPOOLING_H_

#include <Eigen/Core>
#include <vector>
#include <stdexcept>
#include "../Config.h"
#include "../Layer.h"
#include "../Utils/FindMax.h"

namespace MiniDNN {


///
/// \ingroup Layers
///
/// Max-pooling hidden layer
///
/// Currently only supports the "valid" rule of pooling.
///
template <typename Activation>
class MaxPooling: public Layer
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::MatrixXi IntMatrix;

    const int m_channel_rows;
    const int m_channel_cols;
    const int m_in_channels;
    const int m_pool_rows;
    const int m_pool_cols;

    const int m_out_rows;
    const int m_out_cols;

    IntMatrix m_loc;             // Record the locations of maximums
    Matrix m_z;                  // Max pooling results
    Matrix m_a;                  // Output of this layer, a = act(z)
    Matrix m_din;                // Derivative of the input of this layer.
                                 // Note that input of this layer is also the output of previous layer

public:
    // Currently we only implement the "valid" rule
    // https://stackoverflow.com/q/37674306
    ///
    /// Constructor
    ///
    /// \param in_width       Width of the input image in each channel.
    /// \param in_height      Height of the input image in each channel.
    /// \param in_channels    Number of input channels.
    /// \param pooling_width  Width of the pooling window.
    /// \param pooling_height Height of the pooling window.
    ///
    MaxPooling(const int in_width, const int in_height, const int in_channels,
               const int pooling_width, const int pooling_height) :
        Layer(in_width * in_height * in_channels,
              (in_width / pooling_width) * (in_height / pooling_height) * in_channels),
        m_channel_rows(in_height), m_channel_cols(in_width), m_in_channels(in_channels),
        m_pool_rows(pooling_height), m_pool_cols(pooling_width),
        m_out_rows(m_channel_rows / m_pool_rows), m_out_cols(m_channel_cols / m_pool_cols)
    {}

    void init(const Scalar& mu, const Scalar& sigma, RNG& rng) {}

    void forward(const Matrix& prev_layer_data)
    {
        // Each column is an observation
        const int nobs = prev_layer_data.cols();
        m_loc.resize(this->m_out_size, nobs);
        m_z.resize(this->m_out_size, nobs);

        // Use m_loc to store the address of each pooling block relative to the beginning of the data
        int* loc_data = m_loc.data();
        const int channel_end = prev_layer_data.size();
        const int channel_stride = m_channel_rows * m_channel_cols;
        const int col_end_gap = m_channel_rows * m_pool_cols * m_out_cols;
        const int col_stride = m_channel_rows * m_pool_cols;
        const int row_end_gap = m_out_rows * m_pool_rows;
        for(int channel_start = 0; channel_start < channel_end; channel_start += channel_stride)
        {
            const int col_end = channel_start + col_end_gap;
            for(int col_start = channel_start; col_start < col_end; col_start += col_stride)
            {
                const int row_end = col_start + row_end_gap;
                for(int row_start = col_start; row_start < row_end; row_start += m_pool_rows, loc_data++)
                    *loc_data = row_start;
            }
        }

        // Find the location of the max value in each block
        loc_data = m_loc.data();
        const int* const loc_end = loc_data + m_loc.size();
        Scalar* z_data = m_z.data();
        const Scalar* src = prev_layer_data.data();
        for(; loc_data < loc_end; loc_data++, z_data++)
        {
            const int offset = *loc_data;
            *z_data = internal::find_block_max(src + offset, m_pool_rows, m_pool_cols, m_channel_rows, *loc_data);
            *loc_data += offset;
        }

        // Apply activation function
        m_a.resize(this->m_out_size, nobs);
        Activation::activate(m_z, m_a);
    }

    const Matrix& output() const { return m_a; }

    // prev_layer_data: in_size x nobs
    // next_layer_data: out_size x nobs
    void backprop(const Matrix& prev_layer_data, const Matrix& next_layer_data)
    {
        const int nobs = prev_layer_data.cols();

        // After forward stage, m_z contains z = max_pooling(in)
        // Now we need to calculate d(L) / d(z) = [d(a) / d(z)] * [d(L) / d(a)]
        // d(L) / d(z) is computed in the next layer, contained in next_layer_data
        // The Jacobian matrix J = d(a) / d(z) is determined by the activation function
        Matrix& dLz = m_z;
        Activation::apply_jacobian(m_z, m_a, next_layer_data, dLz);

        // d(L) / d(in_i) = sum_j{ [d(z_j) / d(in_i)] * [d(L) / d(z_j)] }
        // d(z_j) / d(in_i) = 1 if in_i is used to compute z_j and is the maximum
        //                  = 0 otherwise
        m_din.resize(this->m_in_size, nobs);
        m_din.setZero();
        const int dLz_size = dLz.size();

        const Scalar* dLz_data = dLz.data();
        const int* loc_data = m_loc.data();
        Scalar* din_data = m_din.data();
        for(int i = 0; i < dLz_size; i++)
            din_data[loc_data[i]] += dLz_data[i];
    }

    const Matrix& backprop_data() const { return m_din; }

    void update(Optimizer& opt) {}

    std::vector<Scalar> get_parameters() const { return std::vector<Scalar>(); }

    void set_parameters(const std::vector<Scalar>& param) {}

    std::vector<Scalar> get_derivatives() const { return std::vector<Scalar>(); }
};


} // namespace MiniDNN


#endif /* LAYER_MAXPOOLING_H_ */
