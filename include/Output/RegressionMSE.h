#ifndef MINIDNN_OUTPUT_REGRESSIONMSE_H_
#define MINIDNN_OUTPUT_REGRESSIONMSE_H_

#include <Eigen/Core>
#include <stdexcept>
#include "../Config.h"
#include "../Output.h"

namespace MiniDNN
{


///
/// \ingroup Outputs
///
/// Regression output layer using Mean Squared Error (MSE) criterion.
///
class RegressionMSE: public Output
{
private:
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    Matrix m_din;  // Derivative of the input of this layer.
                   // Note that input of this layer is also the output of previous layer

public:
    void evaluate(const Matrix& prev_layer_data, const Matrix& target) override
    {
        // Check dimension
        const int nobs = prev_layer_data.cols();
        const int nvar = prev_layer_data.rows();

        if ((target.cols() != nobs) || (target.rows() != nvar))
        {
            throw std::invalid_argument("[class RegressionMSE]: Target data have incorrect dimension");
        }

        // Compute the derivative of the input of this layer
        // L = 0.5 * ||yhat - y||^2 / n
        // in = yhat
        // d(L) / d(in) = (yhat - y) / n
        m_din.resize(nvar, nobs);
        m_din.noalias() = (prev_layer_data - target) / nobs;
    }

    const Matrix& backprop_data() const override
    {
        return m_din;
    }

    Scalar loss() const override
    {
        // L = 0.5 * ||yhat - y||^2 / n
        return Scalar(0.5) * m_din.squaredNorm() * m_din.cols();
    }

    std::string output_type() const override
    {
        return "RegressionMSE";
    }
};


} // namespace MiniDNN


#endif // MINIDNN_OUTPUT_REGRESSIONMSE_H_
