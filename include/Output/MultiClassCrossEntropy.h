#ifndef MINIDNN_OUTPUT_MULTICLASSCROSSENTROPY_H_
#define MINIDNN_OUTPUT_MULTICLASSCROSSENTROPY_H_

#include <Eigen/Core>
#include <stdexcept>
#include "../Config.h"
#include "../Output.h"

namespace MiniDNN
{


///
/// \ingroup Outputs
///
/// Multi-class classification output layer using cross-entropy criterion.
///
class MultiClassCrossEntropy: public Output
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
        // Each column has and only has one 1
        const Index nobs = target.cols();
        const Index nclass = target.rows();

        for (Index i = 0; i < nobs; i++)
        {
            Index one = 0;
            for (Index j = 0; j < nclass; j++)
            {
                if (target(j, i) == Scalar(1))
                {
                    one++;
                    continue;
                }

                if (target(j, i) != Scalar(0))
                {
                    throw std::invalid_argument("[class MultiClassCrossEntropy]: Target data should only contain zero or one");
                }
            }

            if (one != 1)
            {
                throw std::invalid_argument("[class MultiClassCrossEntropy]: Each column of target data should only contain one \"1\"");
            }
        }
    }

    void check_target_data(const IntegerVector& target) override
    {
        // All elements must be non-negative
        const int nobs = target.size();

        for (int i = 0; i < nobs; i++)
        {
            if (target[i] < 0)
            {
                throw std::invalid_argument("[class MultiClassCrossEntropy]: Target data must be non-negative");
            }
        }
    }

    // target is a matrix with each column representing an observation
    // Each column is a vector that has a one at some location and has zeros elsewhere
    void evaluate(const Matrix& prev_layer_data, const Matrix& target) override
    {
        // Check dimension
        const int nobs = prev_layer_data.cols();
        const int nclass = prev_layer_data.rows();

        if ((target.cols() != nobs) || (target.rows() != nclass))
        {
            throw std::invalid_argument("[class MultiClassCrossEntropy]: Target data have incorrect dimension");
        }

        // Compute the derivative of the input of this layer
        // L = -sum(log(phat) * y)
        // in = phat
        // d(L) / d(in) = -y / phat
        m_din.resize(nclass, nobs);
        m_din.noalias() = -target.cwiseQuotient(prev_layer_data);
    }

    // target is a vector of class labels that take values from [0, 1, ..., nclass - 1]
    // The i-th element of target is the class label for observation i
    void evaluate(const Matrix& prev_layer_data, const IntegerVector& target) override
    {
        // Check dimension
        const int nobs = prev_layer_data.cols();
        const int nclass = prev_layer_data.rows();

        if (target.size() != nobs)
        {
            throw std::invalid_argument("[class MultiClassCrossEntropy]: Target data have incorrect dimension");
        }

        // Compute the derivative of the input of this layer
        // L = -log(phat[y])
        // in = phat
        // d(L) / d(in) = [0, 0, ..., -1/phat[y], 0, ..., 0]
        m_din.resize(nclass, nobs);
        m_din.setZero();

        for (int i = 0; i < nobs; i++)
        {
            m_din(target[i], i) = -Scalar(1) / prev_layer_data(target[i], i);
        }
    }

    const Matrix& backprop_data() const override
    {
        return m_din;
    }

    Scalar loss() const override
    {
        // L = -sum(log(phat) * y)
        // in = phat
        // d(L) / d(in) = -y / phat
        // m_din contains 0 if y = 0, and -1/phat if y = 1
        Scalar res = Scalar(0);
        const int nelem = m_din.size();
        const Scalar* din_data = m_din.data();

        for (int i = 0; i < nelem; i++)
        {
            if (din_data[i] < Scalar(0))
            {
                res += std::log(-din_data[i]);
            }
        }

        return res / m_din.cols();
    }

    std::string output_type() const override
    {
        return "MultiClassCrossEntropy";
    }
};


} // namespace MiniDNN


#endif // MINIDNN_OUTPUT_MULTICLASSCROSSENTROPY_H_
