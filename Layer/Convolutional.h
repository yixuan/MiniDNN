#ifndef LAYER_CONVOLUTIONAL_H_
#define LAYER_CONVOLUTIONAL_H_

#include <Eigen/Core>
#include <vector>
#include "../Config.h"
#include "../Layer.h"
#include "../Utils/Convolution.h"
#include "../Utils/Random.h"

template <typename Activation>
class Convolutional: public Layer
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Map<Matrix> MapMat;
    typedef Eigen::Map<const Matrix> ConstMapMat;
    typedef std::vector< std::vector<MapMat> > Tensor4D;
    typedef std::vector< std::vector<ConstMapMat> > ConstTensor4D;
    typedef Matrix::ConstAlignedMapType ConstAlignedMapMat;
    typedef Vector::ConstAlignedMapType ConstAlignedMapVec;
    typedef Vector::AlignedMapType AlignedMapVec;

    const int m_image_rows;   // Number of rows of the input image
    const int m_image_cols;   // Number of columns of the input image
    const int m_filter_rows;  // Number of rows of the filter
    const int m_filter_cols;  // Number of columns of the filter
    const int m_out_rows;     // Number of rows of the output image
    const int m_out_cols;     // Number of columns of the output image
    const int m_in_channels;  // Number of input channels
    const int m_out_channels; // Number of output channels

    Tensor4D m_filter;        // Filter parameters. filter[i][j] is the filter
                              // matrix from in-channel i to out-channel j,
                              // i = 0, 1, ..., in_channels-1
                              // j = 0, 1, ..., out_channels-1
                              // Each filter[i][j] is of size (filter_rows x filter_cols)
    Vector m_filter_data;     // Vector that actually stores the filter data
                              // (in_channels x out_channels x filter_rows x filter_cols)

    Tensor4D m_df;            // Derivative of filters
    Vector m_df_data;         // Vector that actually stores the filter derivatives, same dimension as m_filter_data

    Vector m_bias;            // Bias term for the output channels, out_channels x 1. (One bias term per channel)
    Vector m_db;              // Derivative of bias, same dimension as m_bias

    Matrix m_z;               // Linear term, z = conv(in, w) + b. Each column is an observation
    Matrix m_a;               // Output of this layer, a = act(z)
    Matrix m_din;             // Derivative of the input of this layer.
                              // Note that input of this layer is also the output of previous layer

public:
    Convolutional(const int in_width, const int in_height, const int window_width, const int window_height,
                  const int in_channels, const int out_channels) :
        Layer(in_width * in_height * in_channels,
              (in_width - window_width + 1) * (in_height - window_height + 1) * out_channels),
        m_image_rows(in_height), m_image_cols(in_width), m_filter_rows(window_height), m_filter_cols(window_width),
        m_out_rows(in_height - window_height + 1), m_out_cols(in_width - window_width + 1),
        m_in_channels(in_channels), m_out_channels(out_channels)
    {}

    void init(const Scalar& mu, const Scalar& sigma, RNGType& rng)
    {
        // Set data dimension
        const int filter_data_size = m_in_channels * m_out_channels * m_filter_rows * m_filter_cols;
        m_filter_data.resize(filter_data_size);
        m_df_data.resize(filter_data_size);

        // Create filter tensors
        vector_to_tensor_4d(m_filter_data.data(), m_in_channels, m_out_channels, m_filter_rows, m_filter_cols, m_filter);
        vector_to_tensor_4d(m_df_data.data(),     m_in_channels, m_out_channels, m_filter_rows, m_filter_cols, m_df);

        // Random initialization of filter parameters
        set_normal_random(m_filter_data.data(), filter_data_size, rng, mu, sigma);

        // Bias term
        m_bias.resize(m_out_channels);
        m_db.resize(m_out_channels);
        set_normal_random(m_bias.data(), m_out_channels, rng, mu, sigma);
    }

    // http://cs231n.github.io/convolutional-networks/
    void forward(const Matrix& prev_layer_data)
    {
        // Each column is an observation
        const int nobs = prev_layer_data.cols();

        // Create a tensor for input image
        // 1st index: index of observation
        // 2nd index: index of input channel
        // 3rd and 4th indices: the image matrix
        // For example, in_tensor[i][j] is the image matrix for the j-th input channel of the i-th observation
        ConstTensor4D in_tensor;
        vector_to_tensor_4d(prev_layer_data.data(), nobs, m_in_channels, m_image_rows, m_image_cols, in_tensor);

        // Linear term, z = conv(in, w) + b
        m_z.resize(this->m_out_size, nobs);
        m_z.setZero();

        // Tensor for output image
        // 1st index: observation
        // 2nd index: output channel
        // 3rd and 4th indices: output image matrix
        Tensor4D out_tensor;
        vector_to_tensor_4d(m_z.data(), nobs, m_out_channels, m_out_rows, m_out_cols, out_tensor);

        // For each output channel, z_j = sum_i(conv(in_i, w_ij)) + b_j
        // z_j is an image (matrix), b_j is a scalar
        // The final z stacks z_i together
        //
        // Observation
        for(int k = 0; k < nobs; k++)
        {
            // Ouput channel
            for(int j = 0; j < m_out_channels; j++)
            {
                // Input channel
                for(int i = 0; i < m_in_channels; i++)
                {
                    convolve_valid<Eigen::Dynamic, Eigen::Dynamic>(in_tensor[k][i], m_filter[i][j], out_tensor[k][j]);
                }
                // Add bias term
                out_tensor[k][j].array() += m_bias[j];
            }
        }

        // Apply activation function
        m_a.resize(this->m_out_size, nobs);
        Activation::activate(m_z, m_a);
    }

    const Matrix& output() const
    {
        return m_a;
    }

    // prev: in_size x nobs
    // next: out_size x nobs
    // https://grzegorzgwardys.wordpress.com/2016/04/22/8/
    void backprop(const Matrix& prev_layer_data, const Matrix& next_layer_data)
    {
        const int nobs = prev_layer_data.cols();

        // After forward stage, m_z contains z = conv(in, w) + b
        // Now we need to calculate d_L / d_z = (d_a / d_z) * (d_L / d_a)
        // d_L / d_a is computed in the next layer, contained in next_layer_data
        // The Jacobian matrix J = d_a / d_z is determined by the activation function
        Matrix& dLz = m_z;
        Activation::apply_jacobian(m_z, m_a, next_layer_data, dLz);

        // Tensor for layer input
        ConstTensor4D in_tensor;
        vector_to_tensor_4d(prev_layer_data.data(), nobs, m_in_channels, m_image_rows, m_image_cols, in_tensor);

        // Tensor for d_L / d_z
        ConstTensor4D dLz_tensor;
        vector_to_tensor_4d(dLz.data(), nobs, m_out_channels, m_out_rows, m_out_cols, dLz_tensor);

        // z_j = sum_i(conv(in_i, w_ij)) + b_j
        //
        // d_zk / d_wij = 0, if k != j
        // d_L / d_wij = (d_zj / d_wij) * (d_L / d_zj) = sum_i((d_zj / d_wij) * (d_L / d_zj))
        // = sum_i(conv(in_i, d_L / d_zj))
        //
        // z_j is an image (matrix), b_j is a scalar
        // d_zj / d_bj = a matrix of the same size of d_zj filled with 1
        // d_L / d_bj = (d_L / d_zj).sum()
        //
        // d_zj / d_ini = conv_full_op(w_ij_rotate)
        // d_L / d_ini = sum_j((d_zj / d_ini) * (d_L / d_zj)) = sum_j(conv_full(d_L / d_zj, w_ij_rotate))

        // Derivative for weights
        m_df_data.setZero();
        // Observation
        for(int k = 0; k < nobs; k++)
        {
            // Input channel
            for(int i = 0; i < m_in_channels; i++)
            {
                // Ouput channel
                for(int j = 0; j < m_out_channels; j++)
                {
                    convolve_valid<Eigen::Dynamic, Eigen::Dynamic>(in_tensor[k][i], dLz_tensor[k][j], m_df[i][j]);
                }
            }
        }
        m_df_data /= nobs;

        // Derivative for bias
        // Aggregate d_L / d_z in each output channel
        ConstAlignedMapMat dLz_by_channel(dLz.data(), m_out_rows * m_out_cols, m_out_channels * nobs);
        Vector dLb = dLz_by_channel.colwise().sum();
        // Average over observations
        ConstAlignedMapMat dLb_by_obs(dLb.data(), m_out_channels, nobs);
        m_db.noalias() = dLb_by_obs.rowwise().mean();

        // Tensor for m_din
        m_din.resize(this->m_in_size, nobs);
        m_din.setZero();
        Tensor4D din_tensor;
        vector_to_tensor_4d(m_din.data(), nobs, m_in_channels, m_image_rows, m_image_cols, din_tensor);

        // Compute d_L / d_in = conv_full(d_L / d_z, w_rotate)
        // Observation
        for(int k = 0; k < nobs; k++)
        {
            // Input channel
            for(int i = 0; i < m_in_channels; i++)
            {
                // Ouput channel
                for(int j = 0; j < m_out_channels; j++)
                {
                    convolve_full<Eigen::Dynamic, Eigen::Dynamic>(dLz_tensor[k][j], m_filter[i][j].reverse(), din_tensor[k][i]);
                }
            }
        }
    }

    const Matrix& backprop_data() const
    {
        return m_din;
    }

    void update(Optimizer& opt)
    {
        ConstAlignedMapVec dw(m_df_data.data(), m_df_data.size());
        ConstAlignedMapVec db(m_db.data(), m_db.size());
        AlignedMapVec      w(m_filter_data.data(), m_filter_data.size());
        AlignedMapVec      b(m_bias.data(), m_bias.size());

        opt.update(dw, w);
        opt.update(db, b);
    }

    std::vector<Scalar> get_parameters() const
    {
        std::vector<Scalar> res(m_filter_data.size() + m_bias.size());
        // Copy the data of filters and bias to a long vector
        std::copy(m_filter_data.data(), m_filter_data.data() + m_filter_data.size(), res.begin());
        std::copy(m_bias.data(), m_bias.data() + m_bias.size(), res.begin() + m_filter_data.size());

        return res;
    }

    std::vector<Scalar> get_derivatives() const
    {
        std::vector<Scalar> res(m_df_data.size() + m_db.size());
        // Copy the data of filters and bias to a long vector
        std::copy(m_df_data.data(), m_df_data.data() + m_df_data.size(), res.begin());
        std::copy(m_db.data(), m_db.data() + m_db.size(), res.begin() + m_df_data.size());

        return res;
    }
};


#endif /* LAYER_CONVOLUTIONAL_H_ */
