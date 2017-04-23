#ifndef OPTIMIZER_ADAGRAD_H_
#define OPTIMIZER_ADAGRAD_H_

#include <Eigen/Core>
#include "../Config.h"
#include "../Optimizer.h"

class Adagrad: public Optimizer
{
private:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
	typedef Eigen::Array<Scalar, Eigen::Dynamic, 1> Array;

public:
	void reset()
	{

	}

	void update_vec(const Vector& dvec, Vector& vec)
	{

	}

	void update_mat(const Matrix& dmat, Matrix& mat)
	{

	}
};


#endif /* OPTIMIZER_ADAGRAD_H_ */
