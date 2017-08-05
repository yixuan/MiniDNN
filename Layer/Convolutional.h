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
    typedef std::vector< std::vector<MatWrapper> > Tensor4D;
    typedef std::vector< std::vector<ConstMatWrapper> > ConstTensor4D;
    typedef Matrix::ConstAlignedMapType ConstAlignedMapMat;
    typedef Vector::ConstAlignedMapType ConstAlignedMapVec;
    typedef Vector::AlignedMapType AlignedMapVec;

    const ConvDims m_dim;     // Various dimensions of convolution

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
        m_dim(in_channels, out_channels, in_height, in_width, window_height, window_width)
    {}

    void init(const Scalar& mu, const Scalar& sigma, RNG& rng)
    {
        // Set data dimension
        const int filter_data_size = m_dim.in_channels * m_dim.out_channels * m_dim.filter_rows * m_dim.filter_cols;
        m_filter_data.resize(filter_data_size);
        m_df_data.resize(filter_data_size);

        // Create filter tensors
        vector_to_tensor_4d(m_filter_data.data(), m_dim.in_channels, m_dim.out_channels, m_dim.filter_rows, m_dim.filter_cols, m_filter);
        vector_to_tensor_4d(m_df_data.data(),     m_dim.in_channels, m_dim.out_channels, m_dim.filter_rows, m_dim.filter_cols, m_df);

        // Random initialization of filter parameters
        set_normal_random(m_filter_data.data(), filter_data_size, rng, mu, sigma);

        // Bias term
        m_bias.resize(m_dim.out_channels);
        m_db.resize(m_dim.out_channels);
        set_normal_random(m_bias.data(), m_dim.out_channels, rng, mu, sigma);
    }

    // http://cs231n.github.io/convolutional-networks/
    void forward(const Matrix& prev_layer_data)
    {
        // Each column is an observation
        const int nobs = prev_layer_data.cols();

        // Linear term, z = conv(in, w) + b
        m_z.resize(this->m_out_size, nobs);
        m_z.setZero();
        // Convolution
        convolve_valid(m_dim, prev_layer_data.data(), true, nobs,
            m_filter_data.data(), m_z.data()
        );
        // Add bias terms
        // Each column of m_z contains m_dim.out_channels channels, and each channel has
        // m_dim.conv_rows * m_dim.conv_cols elements
        int channel_start_row = 0;
        const int channel_nelem = m_dim.conv_rows * m_dim.conv_cols;
        for(int i = 0; i < m_dim.out_channels; i++, channel_start_row += channel_nelem)
        {
            m_z.block(channel_start_row, 0, channel_nelem, nobs).array() += m_bias[i];
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

        // Tensor for d_L / d_z
        ConstTensor4D dLz_tensor;
        vector_to_tensor_4d(dLz.data(), nobs, m_dim.out_channels, m_dim.conv_rows, m_dim.conv_cols, dLz_tensor);

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
        ConvDims back_conv_dim(nobs, m_dim.out_channels, m_dim.channel_rows, m_dim.channel_cols, m_dim.conv_rows, m_dim.conv_cols);
        convolve_valid(back_conv_dim, prev_layer_data.data(), false, m_dim.in_channels,
            dLz.data(), m_df_data.data()
        );
        m_df_data /= nobs;

        // Derivative for bias
        // Aggregate d_L / d_z in each output channel
        ConstAlignedMapMat dLz_by_channel(dLz.data(), m_dim.conv_rows * m_dim.conv_cols, m_dim.out_channels * nobs);
        Vector dLb = dLz_by_channel.colwise().sum();
        // Average over observations
        ConstAlignedMapMat dLb_by_obs(dLb.data(), m_dim.out_channels, nobs);
        m_db.noalias() = dLb_by_obs.rowwise().mean();

        // Tensor for m_din
        m_din.resize(this->m_in_size, nobs);
        m_din.setZero();
        Tensor4D din_tensor;
        vector_to_tensor_4d(m_din.data(), nobs, m_dim.in_channels, m_dim.channel_rows, m_dim.channel_cols, din_tensor);

        // Compute d_L / d_in = conv_full(d_L / d_z, w_rotate)
        // Observation
        for(int k = 0; k < nobs; k++)
        {
            // Input channel
            for(int i = 0; i < m_dim.in_channels; i++)
            {
                // Ouput channel
                for(int j = 0; j < m_dim.out_channels; j++)
                {
                    convolve_full(dLz_tensor[k][j], m_filter[i][j], din_tensor[k][i]);
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

    void set_parameters(const std::vector<Scalar>& param)
    {
        if(static_cast<int>(param.size()) != m_filter_data.size() + m_bias.size())
            throw std::invalid_argument("Parameter size does not match");

        std::copy(param.begin(), param.begin() + m_filter_data.size(), m_filter_data.data());
        std::copy(param.begin() + m_filter_data.size(), param.end(), m_bias.data());
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
