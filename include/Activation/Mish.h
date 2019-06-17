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
/// from : https://arxiv.org/abs/1908.08681
///
class Mish
{
    private:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

    public:
        // a = activation(z) = max(z, 0)
        // Z = [z1, ..., zn], A = [a1, ..., an], n observations
        static inline void activate(const Matrix& Z, Matrix& A)
        {
            A.array() = Z.array() * ((((Z.array()).exp()).log1p()).tanh());
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
