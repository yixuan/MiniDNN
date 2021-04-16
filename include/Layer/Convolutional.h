#ifndef MINIDNN_LAYER_CONVOLUTIONAL_H_
#define MINIDNN_LAYER_CONVOLUTIONAL_H_

#include <Eigen/Core>
#include <vector>
#include <stdexcept>
#include "../Config.h"
#include "../Layer.h"
#include "../Utils/Convolution.h"
#include "../Utils/IO.h"
#include "../Utils/Enum.h"

namespace MiniDNN
{


///
/// \ingroup Layers
///
/// Convolutional hidden layer.
///
/// Currently only supports the "valid" rule of convolution.
///
class Convolutional: public Layer
{
private:
    using Index = Eigen::Index;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using ConstAlignedMapMat = Matrix::ConstAlignedMapType;
    using ConstAlignedMapVec = Vector::ConstAlignedMapType;
    using AlignedMapVec = Vector::AlignedMapType;
    using MetaInfo = std::map<std::string, int>;

    using Layer::m_in_size;
    using Layer::m_out_size;

    const internal::ConvDims m_dim; // Various dimensions of convolution

    Vector m_filter_data;  // Filter parameters. Total length is
                           // (in_channels x out_channels x filter_rows x filter_cols)
                           // See Utils/Convolution.h for its layout

    Vector m_df_data;      // Derivative of filters, same dimension as m_filter_data

    Vector m_bias;         // Bias term for the output channels, out_channels x 1. (One bias term per channel)
    Vector m_db;           // Derivative of bias, same dimension as m_bias

    Matrix m_out;          // Linear term, z = conv(in, w) + b. Each column is an observation
    Matrix m_din;          // Derivative of the input of this layer
                           // Note that input of this layer is also the output of previous layer

public:
    ///
    /// Constructor
    ///
    /// \param in_width      Width of the input image in each channel.
    /// \param in_height     Height of the input image in each channel.
    /// \param in_channels   Number of input channels.
    /// \param out_channels  Number of output channels.
    /// \param window_width  Width of the filter.
    /// \param window_height Height of the filter.
    ///
    Convolutional(const int in_width, const int in_height,
                  const int in_channels, const int out_channels,
                  const int window_width, const int window_height) :
        Layer(in_width * in_height * in_channels,
              (in_width - window_width + 1) * (in_height - window_height + 1) * out_channels),
        m_dim(in_channels, out_channels, in_height, in_width, window_height, window_width)
    {}

    void init(const Initializer& initializer, RNG& rng) override
    {
        // Set data dimension
        init();
        // Set random coefficients
        initializer.initialize(m_filter_data, rng);
        initializer.initialize(m_bias, rng);
    }

    void init() override
    {
        // Set parameter dimension
        const Index filter_data_size = m_dim.in_channels * m_dim.out_channels *
                                       m_dim.filter_rows * m_dim.filter_cols;
        // Filter parameters
        m_filter_data.resize(filter_data_size);
        m_df_data.resize(filter_data_size);
        // Bias term
        m_bias.resize(m_dim.out_channels);
        m_db.resize(m_dim.out_channels);
    }

    // http://cs231n.github.io/convolutional-networks/
    void forward(const Matrix& prev_layer_data) override
    {
        // Each column is an observation
        const int nobs = prev_layer_data.cols();
        // Linear term, z = conv(in, w) + b
        m_out.resize(m_out_size, nobs);
        // Convolution
        internal::convolve_valid(m_dim, prev_layer_data.data(), true, nobs,
                                 m_filter_data.data(), m_out.data());
        // Add bias terms
        // Each column of m_out contains m_dim.out_channels channels, and each channel has
        // m_dim.conv_rows * m_dim.conv_cols elements
        Index channel_start_row = 0;
        const Index channel_nelem = m_dim.conv_rows * m_dim.conv_cols;
        for (Index i = 0; i < m_dim.out_channels; i++, channel_start_row += channel_nelem)
        {
            m_out.block(channel_start_row, 0, channel_nelem, nobs).array() += m_bias[i];
        }
    }

    const Matrix& output() const override
    {
        return m_out;
    }

    // prev_layer_data: in_size x nobs
    // next_layer_data: out_size x nobs
    // https://grzegorzgwardys.wordpress.com/2016/04/22/8/
    void backprop(const Matrix& prev_layer_data, const Matrix& next_layer_data) override
    {
        const int nobs = prev_layer_data.cols();
        const Matrix& dldz = next_layer_data;
        // prev_layer_data [in_size x n]  => in
        // m_out           [out_size x n] => Z = W' * in + b
        // next_layer_data [out_size x n] => dl/dZ
        //
        // z_j = sum_i(conv(in_i, w_ij)) + b_j
        //
        // d(z_k) / d(w_ij) = 0, if k != j
        // d(L) / d(w_ij) = [d(z_j) / d(w_ij)] * [d(L) / d(z_j)] = sum_i{ [d(z_j) / d(w_ij)] * [d(L) / d(z_j)] }
        // = sum_i(conv(in_i, d(L) / d(z_j)))
        //
        // z_j is an image (matrix), b_j is a scalar
        // d(z_j) / d(b_j) = a matrix of the same size of d(z_j) filled with 1
        // d(L) / d(b_j) = (d(L) / d(z_j)).sum()
        //
        // d(z_j) / d(in_i) = conv_full_op(w_ij_rotate)
        // d(L) / d(in_i) = sum_j((d(z_j) / d(in_i)) * (d(L) / d(z_j))) = sum_j(conv_full(d(L) / d(z_j), w_ij_rotate))
        //
        // Derivative for weights
        internal::ConvDims back_conv_dim(nobs, m_dim.out_channels, m_dim.channel_rows, m_dim.channel_cols,
                                         m_dim.conv_rows, m_dim.conv_cols);
        internal::convolve_valid(back_conv_dim, prev_layer_data.data(), false,
                                 m_dim.in_channels,
                                 dldz.data(), m_df_data.data());
        // Derivative for bias
        // Aggregate d(L) / d(z) in each output channel
        ConstAlignedMapMat dldz_by_channel(dldz.data(), m_dim.conv_rows * m_dim.conv_cols,
                                           m_dim.out_channels * nobs);
        Vector dldb = dldz_by_channel.colwise().sum();
        // Sum over observations
        ConstAlignedMapMat dldb_by_obs(dldb.data(), m_dim.out_channels, nobs);
        m_db.noalias() = dldb_by_obs.rowwise().sum();
        // Compute d(L) / d_in = conv_full(d(L) / d(z), w_rotate)
        m_din.resize(m_in_size, nobs);
        internal::ConvDims conv_full_dim(m_dim.out_channels, m_dim.in_channels,
                                         m_dim.conv_rows, m_dim.conv_cols,
                                         m_dim.filter_rows, m_dim.filter_cols);
        internal::convolve_full(conv_full_dim, dldz.data(), nobs,
                                m_filter_data.data(), m_din.data());
    }

    const Matrix& backprop_data() const override
    {
        return m_din;
    }

    void update(Optimizer& opt) override
    {
        ConstAlignedMapVec dw(m_df_data.data(), m_df_data.size());
        ConstAlignedMapVec db(m_db.data(), m_db.size());
        AlignedMapVec      w(m_filter_data.data(), m_filter_data.size());
        AlignedMapVec      b(m_bias.data(), m_bias.size());
        opt.update(dw, w);
        opt.update(db, b);
    }

    std::vector<Scalar> get_parameters() const override
    {
        std::vector<Scalar> res(m_filter_data.size() + m_bias.size());
        // Copy the data of filters and bias to a long vector
        std::copy(m_filter_data.data(), m_filter_data.data() + m_filter_data.size(), res.begin());
        std::copy(m_bias.data(), m_bias.data() + m_bias.size(), res.begin() + m_filter_data.size());
        return res;
    }

    void set_parameters(const std::vector<Scalar>& param) override
    {
        if (static_cast<int>(param.size()) != m_filter_data.size() + m_bias.size())
        {
            throw std::invalid_argument("[class Convolutional]: Parameter size does not match");
        }

        std::copy(param.begin(), param.begin() + m_filter_data.size(), m_filter_data.data());
        std::copy(param.begin() + m_filter_data.size(), param.end(), m_bias.data());
    }

    std::vector<Scalar> get_derivatives() const override
    {
        std::vector<Scalar> res(m_df_data.size() + m_db.size());
        // Copy the data of filters and biases to a long vector
        std::copy(m_df_data.data(), m_df_data.data() + m_df_data.size(), res.begin());
        std::copy(m_db.data(), m_db.data() + m_db.size(), res.begin() + m_df_data.size());
        return res;
    }

    std::string layer_type() const override
    {
        return "Convolutional";
    }

    void fill_meta_info(MetaInfo& map, int index) const override
    {
        std::string ind = std::to_string(index);
        const int layerid = static_cast<int>(internal::layer_id(layer_type()));
        map.insert(std::make_pair("Layer" + ind, layerid));
        map.insert(std::make_pair("in_channels" + ind, m_dim.in_channels));
        map.insert(std::make_pair("out_channels" + ind, m_dim.out_channels));
        map.insert(std::make_pair("in_height" + ind, m_dim.channel_rows));
        map.insert(std::make_pair("in_width" + ind, m_dim.channel_cols));
        map.insert(std::make_pair("window_width" + ind, m_dim.filter_cols));
        map.insert(std::make_pair("window_height" + ind, m_dim.filter_rows));
    }
};


} // namespace MiniDNN


#endif // MINIDNN_LAYER_CONVOLUTIONAL_H_
