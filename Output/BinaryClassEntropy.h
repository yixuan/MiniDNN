#ifndef OUTPUT_BINARYCLASSENTROPY_H_
#define OUTPUT_BINARYCLASSENTROPY_H_

#include <Eigen/Core>
#include <exception>
#include "../Config.h"

class BinaryClassEntropy: public Output
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

    Matrix m_din;  // Derivative of the input of this layer.
                   // Note that input of this layer is also the output of previous layer

public:
    void evaluate(const Matrix& prev_layer_data, const Matrix& target)
    {
        // Check dimension
        const int nobs = prev_layer_data.cols();
        const int nvar = prev_layer_data.rows();
        if((target.cols() != nobs) || (target.rows() != nvar))
            throw std::domain_error("Target data have incorrect dimension");

        // Compute the derivative of the input of this layer
        // L = -y * log(phat) - (1 - y) * log(1 - phat)
        // in = phat
        // d_L / d_in = -y / phat + (1 - y) / (1 - phat), y is either 0 or 1
        m_din.resize(nvar, nobs);
        m_din.array() = (target.array() < Scalar(0.5)).select((Scalar(1) - prev_layer_data.array()).cwiseInverse(),
                                                              -prev_layer_data.cwiseInverse());
    }

    const Matrix& backprop_data() const
    {
        return m_din;
    }

    Scalar loss(const Matrix& prev_layer_data, const Matrix& target) const
    {
        // Dimension has been checked in evaluate()
        const int nobs = prev_layer_data.cols();

        // L = -y * log(phat) - (1 - y) * log(1 - phat)
        // y = 0 => L = -log(1 - phat)
        // y = 1 => L = -log(phat)
        // m_din contains 1/(1 - phat) if y = 0, and -1/phat if y = 1, so
        // L = log(abs(m_din)).sum()
        return m_din.array().abs().log().sum() / nobs;
    }
};


#endif /* OUTPUT_BINARYCLASSENTROPY_H_ */
