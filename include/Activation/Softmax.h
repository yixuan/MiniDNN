#ifndef MINIDNN_ACTIVATION_SOFTMAX_H_
#define MINIDNN_ACTIVATION_SOFTMAX_H_

namespace MiniDNN
{


///
/// \ingroup Activations
///
/// The softmax activation function.
///
class Softmax: public Activation
{
private:
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using RowArray = Eigen::Array<Scalar, 1, Eigen::Dynamic>;
    using Activation::m_out;
    using Activation::m_din;

public:
    // a = f(z) = softmax(z)
    // Z = [z1, ..., zn], A = [a1, ..., an], n observations
    // Z => prev_layer_data [d x n]
    // A => m_out [d x n]
    void forward(const Matrix& prev_layer_data) override
    {
        // Alias for brevity
        const Matrix& z = prev_layer_data;
        m_out.resize(z.rows(), z.cols());
        m_out.array() = z.array().cwiseMax(Scalar(0));

        m_out.array() = (z.rowwise() - z.colwise().maxCoeff()).array().exp();
        RowArray colsums = m_out.colwise().sum();
        m_out.array().rowwise() /= colsums;
    }

    // dl/dz = J * dl/da, for each vector pair (z, a)
    // J = da/dz = diag(a) - a * a'
    // J * dl/da = a .* dl/da - a * (a' * dl/da) = a .* (dl/da - a'(dl/da))
    // dl/dZ = dl/dA .* (dl/dA - [a'(dl/da)])
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

        RowArray a_dot_dlda = a.cwiseProduct(dlda).colwise().sum();
        m_din.array() = a.array() * (dlda.array().rowwise() - a_dot_dlda);
    }

    std::string layer_type() const override
    {
        return "Softmax";
    }
};


} // namespace MiniDNN


#endif // MINIDNN_ACTIVATION_SOFTMAX_H_
