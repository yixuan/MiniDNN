#ifndef ACTIVATION_RELU_H_
#define ACTIVATION_RELU_H_

#include <Eigen/Core>
#include "../Config.h"

class ReLU
{
private:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

public:
	// dest = act(src)
	static inline void activate(const Matrix& src, Matrix& dest)
	{
		dest.array() = src.array().cwiseMax(Scalar(0));
	}

	// src_act = act(src)
	// dest = act'(src)
	static inline void deriv_activate(const Matrix& src, const Matrix& src_act, Matrix& dest)
	{
		dest.array() = src_act.array().cwiseSign();
	}
};

#endif /* ACTIVATION_RELU_H_ */
