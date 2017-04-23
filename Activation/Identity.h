#ifndef ACTIVATION_IDENTITY_H_
#define ACTIVATION_IDENTITY_H_

#include <Eigen/Core>
#include "../Config.h"

class Identity
{
private:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

public:
	// dest = act(src)
	static inline void activate(const Matrix& src, Matrix& dest)
	{
		dest.noalias() = src;
	}

	// src_act = act(src)
	// dest = act'(src)
	static inline void deriv_activate(const Matrix& src, const Matrix& src_act, Matrix& dest)
	{
		dest.setOnes();
	}
};


#endif /* ACTIVATION_IDENTITY_H_ */
