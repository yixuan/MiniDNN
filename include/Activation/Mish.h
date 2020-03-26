#ifndef ACTIVATION_MISH_H_
#define ACTIVATION_MISH_H_

#include <Eigen/Core>
#include "../Config.h"

namespace MiniDNN
{


///
/// \ingroup Activations
///
/// The Mish activation function
///
/// From: https://arxiv.org/abs/1908.08681
///
class Mish
{
    private:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

    public:
        // Mish(x) = x * tanh(softplus(x))
        // softplus(x) = log(1 + exp(x))
        // a = activation(z) = Mish(z)
        // Z = [z1, ..., zn], A = [a1, ..., an], n observations
        static inline void activate(const Matrix& Z, Matrix& A)
        {
            // h(x) = tanh(softplus(x)) = (1 + exp(x))^2 - 1
            //                            ------------------
            //                            (1 + exp(x))^2 + 1
            // Let s = exp(-abs(x)), t = 1 + s
            // If x >= 0, then h(x) = (t^2 - s^2) / (t^2 + s^2)
            // If x <= 0, then h(x) = (t^2 - 1) / (t^2 + 1)
            Matrix S = (-Z.array().abs()).exp();
            A.array() = (S.array() + Scalar(1)).square();  // t^2
            S.noalias() = (Z.array() >= Scalar(0)).select(S.cwiseAbs2(), Scalar(1));  // s^2 or 1
            A.array() = (A.array() - S.array()) / (A.array() + S.array());
            A.array() *= Z.array();
        }

        // Apply the Jacobian matrix J to a vector f
        // J = d_a / d_z = diag(sign(a)) = diag(a > 0)
        // g = J * f = (a > 0) .* f
        // Z = [z1, ..., zn], G = [g1, ..., gn], F = [f1, ..., fn]
        // Note: When entering this function, Z and G may point to the same matrix
        static inline void apply_jacobian(const Matrix& Z, const Matrix& A,
                                          const Matrix& F, Matrix& G)
        {
            Matrix tempSoftplus;
            Matrix tempSech;
            Matrix ex;
            ex.array() = Z.array().exp();
            tempSoftplus.array() = ex.array().log1p();
            tempSech.array() = Scalar(1) / (tempSoftplus.array().cosh());
            G.array() = tempSoftplus.array().tanh() + Z.array() * ex.array() *
                        tempSech.array() * (tempSech.array() / (Scalar(1) + ex.array())) * F.array();
        }

        static std::string return_type()
        {
            return "Mish";
        }
};


} // namespace MiniDNN


#endif /* ACTIVATION_MISH_H_ */
