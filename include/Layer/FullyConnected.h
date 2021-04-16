#ifndef MINIDNN_LAYER_FULLYCONNECTED_H_
#define MINIDNN_LAYER_FULLYCONNECTED_H_

#include <Eigen/Core>
#include <vector>
#include <stdexcept>
#include "../Config.h"
#include "../Layer.h"

namespace MiniDNN
{


///
/// \ingroup Layers
///
/// Fully connected hidden layer.
///
class FullyConnected: public Layer
{
private:
    using Index = Eigen::Index;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using ConstAlignedMapVec = Vector::ConstAlignedMapType;
    using AlignedMapVec = Vector::AlignedMapType;
    using MetaInfo = std::map<std::string, int>;

    using Layer::m_in_size;
    using Layer::m_out_size;

    Matrix m_weight;  // Weight parameters, W [in_size x out_size]
    Vector m_bias;    // Bias parameters, b [out_size x 1]
    Matrix m_dw;      // Derivative of weights
    Vector m_db;      // Derivative of bias
    Matrix m_out;     // Linear term, z = W' * in + b
    Matrix m_din;     // Derivative of the input of this layer.
                      // Note that input of this layer is also the output of previous layer

public:
    ///
    /// Constructor
    ///
    /// \param in_size  Number of input units.
    /// \param out_size Number of output units.
    ///
    FullyConnected(const int in_size, const int out_size) :
        Layer(in_size, out_size)
    {}

    void init(const Initializer& initializer, RNG& rng) override
    {
        // Set parameter dimension
        init();
        // Set random coefficients
        initializer.initialize(m_weight, rng);
        initializer.initialize(m_bias, rng);
    }

    void init() override
    {
        // Set parameter dimension
        m_weight.resize(m_in_size, m_out_size);
        m_bias.resize(m_out_size);
        m_dw.resize(m_in_size, m_out_size);
        m_db.resize(m_out_size);
    }

    // prev_layer_data: in_size x nobs
    void forward(const Matrix& prev_layer_data) override
    {
        const int nobs = prev_layer_data.cols();
        // Linear term z = W' * in + b
        m_out.resize(m_out_size, nobs);
        m_out.noalias() = m_weight.transpose() * prev_layer_data;
        m_out.colwise() += m_bias;
    }

    const Matrix& output() const override
    {
        return m_out;
    }

    // prev_layer_data: in_size x nobs
    // next_layer_data: out_size x nobs
    void backprop(const Matrix& prev_layer_data, const Matrix& next_layer_data) override
    {
        const int nobs = prev_layer_data.cols();
        // prev_layer_data [in_size x n]  => in
        // m_out           [out_size x n] => Z = W' * in + b
        // next_layer_data [out_size x n] => dl/dZ
        //
        // dl/dW [in_size x out_size] = in * (dl/dZ)'
        // dl/db [out_size x 1] = (dl/dZ) * 1_n
        // dl/din [in_size x n] = W * (dl/dZ)
        m_dw.noalias() = prev_layer_data * next_layer_data.transpose();
        m_db.noalias() = next_layer_data.rowwise().sum();
        m_din.resize(m_in_size, nobs);
        m_din.noalias() = m_weight * next_layer_data;
    }

    const Matrix& backprop_data() const override
    {
        return m_din;
    }

    void update(Optimizer& opt) override
    {
        ConstAlignedMapVec dw(m_dw.data(), m_dw.size());
        ConstAlignedMapVec db(m_db.data(), m_db.size());
        AlignedMapVec      w(m_weight.data(), m_weight.size());
        AlignedMapVec      b(m_bias.data(), m_bias.size());
        opt.update(dw, w);
        opt.update(db, b);
    }

    std::vector<Scalar> get_parameters() const override
    {
        std::vector<Scalar> res(m_weight.size() + m_bias.size());
        // Copy the data of weights and bias to a long vector
        std::copy(m_weight.data(), m_weight.data() + m_weight.size(), res.begin());
        std::copy(m_bias.data(), m_bias.data() + m_bias.size(), res.begin() + m_weight.size());
        return res;
    }

    void set_parameters(const std::vector<Scalar>& param) override
    {
        if (static_cast<Index>(param.size()) != m_weight.size() + m_bias.size())
        {
            throw std::invalid_argument("[class FullyConnected]: Parameter size does not match");
        }

        std::copy(param.begin(), param.begin() + m_weight.size(), m_weight.data());
        std::copy(param.begin() + m_weight.size(), param.end(), m_bias.data());
    }

    std::vector<Scalar> get_derivatives() const override
    {
        std::vector<Scalar> res(m_dw.size() + m_db.size());
        // Copy the data of weights and biases to a long vector
        std::copy(m_dw.data(), m_dw.data() + m_dw.size(), res.begin());
        std::copy(m_db.data(), m_db.data() + m_db.size(), res.begin() + m_dw.size());
        return res;
    }

    std::string layer_type() const override
    {
        return "FullyConnected";
    }
};


} // namespace MiniDNN


#endif // MINIDNN_LAYER_FULLYCONNECTED_H_
