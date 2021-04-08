#ifndef MINIDNN_ACTIVATION_SIGMOID_H_
#define MINIDNN_ACTIVATION_SIGMOID_H_

#include <Eigen/Core>
#include "../Config.h"
#include "../Activation.h"

namespace MiniDNN
{


///
/// \ingroup Activations
///
/// The sigmoid activation function.
///
class Sigmoid: public Activation
{
private:
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Activation::m_out;
    using Activation::m_din;

public:
    // a = f(z) = 1 / (1 + exp(-z))
    // Z = [z1, ..., zn], A = [a1, ..., an], n observations
    // Z => prev_layer_data [d x n]
    // A => m_out [d x n]
    void forward(const Matrix& prev_layer_data) override
    {
        // Alias for brevity
        const Matrix& z = prev_layer_data;
        m_out.resize(z.rows(), z.cols());
        m_out.array() = Scalar(1) / (Scalar(1) + (-z.array()).exp());
    }

    // dl/dZ = dl/dA .* f'(Z)
    // f'(z) = a * (1 - a), a = f(z)
    // Z     => prev_layer_data [d x n]
    // dl/dA => next_layer_data [d x n]
    // dl/dZ => m_din [d x n]
    void backprop(const Matrix& prev_layer_data, const Matrix& next_layer_data) override
    {
        // Aliases for brevity
        const Matrix& z = prev_layer_data;
        const Matrix& a = m_out;
        const Matrix& dlda = next_layer_data;
        m_din.resize(dlda.rows(), dlda.cols());
        m_din.array() = a.array() * (Scalar(1) - a.array()) * dlda.array();
    }

    std::string layer_type() const override
    {
        return "Sigmoid";
    }
};


} // namespace MiniDNN


#endif // MINIDNN_ACTIVATION_SIGMOID_H_
