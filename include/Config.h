#ifndef CONFIG_H_
#define CONFIG_H_

namespace MiniDNN
{

// Floating-point number type
#ifndef MDNN_SCALAR
typedef double Scalar;
#else
typedef MDNN_SCALAR Scalar;
#endif

#if(MDNN_ROWMAJOR == 1)
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;
#else
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
#endif

typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;


} // namespace MiniDNN

#endif /* CONFIG_H_ */
