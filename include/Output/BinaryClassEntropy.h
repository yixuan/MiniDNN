#ifndef MNIST_OUTPUT_BINARYCLASSENTROPY_H_
#define MNIST_OUTPUT_BINARYCLASSENTROPY_H_

#include <Eigen/Core>
#include <stdexcept>
#include "../Config.h"
#include "../Output.h"

namespace MiniDNN
{


///
/// \ingroup Outputs
///
/// Binary classification output layer using cross-entropy criterion.
///
class BinaryClassEntropy: public Output
{
private:
    using Index = Eigen::Index;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using IntegerVector = Eigen::RowVectorXi;

    Matrix m_din;  // Derivative of the input of this layer.
                   // Note that input of this layer is also the output of previous layer

public:
    void check_target_data(const Matrix& target) override
    {
        // Each element should be either 0 or 1
        const Index nelem = target.size();
        const Scalar* target_data = target.data();

        for (Index i = 0; i < nelem; i++)
        {
            if ((target_data[i] != Scalar(0)) && (target_data[i] != Scalar(1)))
            {
                throw std::invalid_argument("[class BinaryClassEntropy]: Target data should only contain zero or one");
            }
        }
    }

    void check_target_data(const IntegerVector& target) override
    {
        // Each element should be either 0 or 1
        const Index nobs = target.size();

        for (Index i = 0; i < nobs; i++)
        {
            if ((target[i] != 0) && (target[i] != 1))
            {
                throw std::invalid_argument("[class BinaryClassEntropy]: Target data should only contain zero or one");
            }
        }
    }

    void evaluate(const Matrix& prev_layer_data, const Matrix& target) override
    {
        // Check dimension
        const Index nobs = prev_layer_data.cols();
        const Index nvar = prev_layer_data.rows();
        const Index nelem = nobs * nvar;

        if ((target.cols() != nobs) || (target.rows() != nvar))
        {
            throw std::invalid_argument("[class BinaryClassEntropy]: Target data have incorrect dimension");
        }

        // Compute the derivative of the input of this layer
        // L = -y * log(phat) - (1 - y) * log(1 - phat)
        // in = phat
        // d（L） / d（in） = -y / phat + (1 - y) / (1 - phat), y is either 0 or 1
        m_din.resize(nvar, nobs);
        const Scalar* y_data = target.data();
        const Scalar* phat_data = prev_layer_data.data();
        Scalar* din_data = m_din.data();
        for (Index i = 0; i < nelem; i++)
        {
            din_data[i] = (y_data[i] < Scalar(0.5)) ?
                          (Scalar(1) / (Scalar(1) - phat_data[i])) :
                          (-Scalar(1) / phat_data[i]);
        }
    }

    void evaluate(const Matrix& prev_layer_data, const IntegerVector& target) override
    {
        // Only when the last hidden layer has only one unit can we use this version
        const Index nvar = prev_layer_data.rows();

        if (nvar != 1)
        {
            throw std::invalid_argument("[class BinaryClassEntropy]: Only one response variable is allowed when class labels are used as target data");
        }

        // Check dimension
        const Index nobs = prev_layer_data.cols();

        if (target.size() != nobs)
        {
            throw std::invalid_argument("[class BinaryClassEntropy]: Target data have incorrect dimension");
        }

        // Same as above
        m_din.resize(1, nobs);
        const int* y_data = target.data();
        const Scalar* phat_data = prev_layer_data.data();
        Scalar* din_data = m_din.data();
        for (Index i = 0; i < nobs; i++)
        {
            din_data[i] = (y_data[i] == 0) ?
                          (Scalar(1) / (Scalar(1) - phat_data[i])) :
                          (-Scalar(1) / phat_data[i]);
        }
    }

    const Matrix& backprop_data() const override
    {
        return m_din;
    }

    Scalar loss() const override
    {
        // L = -y * log(phat) - (1 - y) * log(1 - phat)
        // y = 0 => L = -log(1 - phat)
        // y = 1 => L = -log(phat)
        // m_din contains 1/(1 - phat) if y = 0, and -1/phat if y = 1, so
        // L = log(abs(m_din)).sum()
        return m_din.array().abs().log().sum() / m_din.cols();
    }

    std::string output_type() const override
    {
        return "BinaryClassEntropy";
    }
};


} // namespace MiniDNN


#endif // MNIST_OUTPUT_BINARYCLASSENTROPY_H_
