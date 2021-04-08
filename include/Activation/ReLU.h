#ifndef MINIDNN_ACTIVATION_RELU_H_
#define MINIDNN_ACTIVATION_RELU_H_

#include <Eigen/Core>
#include "../Config.h"
#include "../Activation.h"

namespace MiniDNN
{


///
/// \ingroup Activations
///
/// The ReLU activation function.
///
class ReLU: public Activation
{
private:
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Activation::m_out;
    using Activation::m_din;

public:
    // a = f(z) = max(z, 0)
    // Z = [z1, ..., zn], A = [a1, ..., an], n observations
    // Z => prev_layer_data [d x n]
    // A => m_out [d x n]
    void forward(const Matrix& prev_layer_data) override
    {
        // Alias for brevity
        const Matrix& z = prev_layer_data;
        m_out.resize(z.rows(), z.cols());
        m_out.array() = z.array().cwiseMax(Scalar(0));
    }

    // dl/dZ = dl/dA .* f'(Z)
    // f'(z) = 0, if z <= 0
    //         1, if z > 0
    // Z     => prev_layer_data [d x n]
    // dl/dA => next_layer_data [d x n]
    // dl/dZ => m_din [d x n]
    void backprop(const Matrix& prev_layer_data, const Matrix& next_layer_data) override
    {
        // Aliases for brevity
        const Matrix& z = prev_layer_data;
        const Matrix& dlda = next_layer_data;
        m_din.resize(dlda.rows(), dlda.cols());
        m_din.array() = (z.array() > Scalar(0)).select(dlda, Scalar(0));
    }

    std::string layer_type() const override
    {
        return "ReLU";
    }
};


} // namespace MiniDNN


#endif // MINIDNN_ACTIVATION_RELU_H_
