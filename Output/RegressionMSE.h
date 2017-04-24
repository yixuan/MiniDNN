#ifndef OUTPUT_REGRESSIONMSE_H_
#define OUTPUT_REGRESSIONMSE_H_

#include <Eigen/Core>
#include <stdexcept>
#include "../Config.h"

class RegressionMSE: public Output
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

    Matrix m_din;  // Derivative of the input of this layer.
                   // Note that input of this layer is also the output of previous layer

public:
    void check_target_data(const IntegerVector& target)
    {
        // This version is not supported by regression models
        // We simply raise an exception here
        throw std::invalid_argument("Regression models do not support target data as class labels");
    }

    void evaluate(const Matrix& prev_layer_data, const Matrix& target)
    {
        // Check dimension
        const int nobs = prev_layer_data.cols();
        const int nvar = prev_layer_data.rows();
        if((target.cols() != nobs) || (target.rows() != nvar))
            throw std::invalid_argument("Target data have incorrect dimension");

        // Compute the derivative of the input of this layer
        // L = 0.5 * ||yhat - y||^2
        // in = yhat
        // d_L / d_in = yhat - y
        m_din.resize(nvar, nobs);
        m_din.noalias() = prev_layer_data - target;
    }

    void evaluate(const Matrix& prev_layer_data, const IntegerVector& target)
    {
        // This version is not supported by regression models
        // We simply raise an exception here
        throw std::invalid_argument("Regression models do not support target data as class labels");
    }

    const Matrix& backprop_data() const
    {
        return m_din;
    }

    Scalar loss(const Matrix& prev_layer_data, const Matrix& target) const
    {
        // Dimension has been checked in evaluate()
        const int nobs = prev_layer_data.cols();

        // L = 0.5 * ||yhat - y||^2
        return m_din.squaredNorm() / nobs * Scalar(0.5);
    }

    Scalar loss(const Matrix& prev_layer_data, const IntegerVector& target) const
    {
        // This version is not supported by regression models
        // We simply raise an exception here
        throw std::invalid_argument("Regression models do not support target data as class labels");
    }
};


#endif /* OUTPUT_REGRESSIONMSE_H_ */
