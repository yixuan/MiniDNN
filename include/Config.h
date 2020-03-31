#ifndef CONFIG_H_
#define CONFIG_H_

namespace MiniDNN
{

typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

// Floating-point number type
#ifndef MDNN_SCALAR
typedef double Scalar;
#else
typedef MDNN_SCALAR Scalar;
#endif


} // namespace MiniDNN

#endif /* CONFIG_H_ */
