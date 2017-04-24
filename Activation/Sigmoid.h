#ifndef ACTIVATION_SIGMOID_H_
#define ACTIVATION_SIGMOID_H_

#include <Eigen/Core>
#include "../Config.h"

class Sigmoid
{
private:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

public:
	// dest = act(src)
	static inline void activate(const Matrix& src, Matrix& dest)
	{
		dest.array() = Scalar(1) / (Scalar(1) + (-src.array()).exp());
	}

	// src_act = act(src)
	// dest = act'(src)
	static inline void deriv_activate(const Matrix& src, const Matrix& src_act, Matrix& dest)
	{
		dest.array() = src_act.array() * (Scalar(1) - src_act.array());
	}
};


#endif /* ACTIVATION_SIGMOID_H_ */
