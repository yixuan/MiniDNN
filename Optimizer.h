#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include <Eigen/Core>
#include "Config.h"

class Optimizer
{
protected:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

public:
	virtual ~Optimizer() {}

	virtual void reset() = 0;

	virtual void update_vec(const Vector& dvec, Vector& vec) = 0;

	virtual void update_mat(const Matrix& dmat, Matrix& mat) = 0;
};


#endif /* OPTIMIZER_H_ */
