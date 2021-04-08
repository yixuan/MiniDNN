#ifndef MINIDNN_ACTIVATION_MISH_H_
#define MINIDNN_ACTIVATION_MISH_H_

#include <Eigen/Core>
#include "../Config.h"
#include "../Activation.h"

namespace MiniDNN
{


///
/// \ingroup Activations
///
/// The Mish activation function.
///
/// From: https://arxiv.org/abs/1908.08681.
///
class Mish: public Activation
{
private:
    using Index = Eigen::Index;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Activation::m_out;
    using Activation::m_din;

public:
    // a = f(z) = Mish(z) = z * tanh(softplus(z)), softplus(x) = log(1 + exp(x))
    // Z = [z1, ..., zn], A = [a1, ..., an], n observations
    // Z => prev_layer_data [d x n]
    // A => m_out [d x n]
    void forward(const Matrix& prev_layer_data) override
    {
        // Alias for brevity
        const Matrix& z = prev_layer_data;
        m_out.resize(z.rows(), z.cols());

        // h(x) = tanh(softplus(x)) = (1 + exp(x))^2 - 1
        //                            ------------------
        //                            (1 + exp(x))^2 + 1
        // Let s = exp(-abs(x)), t = 1 + s
        // If x >= 0, then h(x) = (t^2 - s^2) / (t^2 + s^2)
        // If x <= 0, then h(x) = (t^2 - 1) / (t^2 + 1)

        const Scalar* zptr = z.data();
        Scalar* aptr = m_out.data();
        Index len = m_out.size();
        for (Index i = 0; i < len; i++)
        {
            const Scalar x = zptr[i];
            const Scalar s = std::exp(-std::abs(x));
            const Scalar t = Scalar(1) + s;
            const Scalar t2 = t * t;
            const Scalar c = (x >= Scalar(0)) ? (s * s) : Scalar(1);  // s^2 or 1
            aptr[i] = x * (t2 - c) / (t2 + c);
        }
    }

    // dl/dZ = dl/dA .* f'(Z)
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

        // Let h(x) = tanh(softplus(x))
        // f'(x) = h(x) + x * h'(x)
        // h'(x) = tanh'(softplus(x)) * softplus'(x)
        //       = [1 - h(x)^2] * exp(x) / (1 + exp(x))
        //       = [1 - h(x)^2] / (1 + exp(-x))
        // f'(x) = h(x) + [x - f(x) * h(x)] / (1 + exp(-x))
        // a = f(z) = z * h(z) => h(z) = a / z, h(0) = 0.6

        const Scalar* zptr = z.data();
        const Scalar* dldaptr = dlda.data();
        const Scalar* aptr = a.data();
        Scalar* dldzptr = m_din.data();
        Index len = m_out.size();
        for (Index i = 0; i < len; i++)
        {
            const Scalar x = zptr[i];
            const Scalar h = (x == Scalar(0)) ? Scalar(0.6) : (aptr[i] / x);
            const Scalar dh = (Scalar(1) - h * h) / (Scalar(1) + std::exp(-x));
            const Scalar df = h + x * dh;
            dldzptr[i] = dldaptr[i] * df;
        }
    }

    std::string layer_type() const override
    {
        return "Mish";
    }
};


} // namespace MiniDNN


#endif // MINIDNN_ACTIVATION_MISH_H_
