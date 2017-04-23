#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include <Eigen/Core>
#include "Config.h"

class Optimizer
{
protected:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
	typedef Vector::ConstAlignedMapType ConstAlignedMapVec;
	typedef Vector::AlignedMapType AlignedMapVec;

public:
	virtual ~Optimizer() {}

	virtual void reset() = 0;

	virtual void update(ConstAlignedMapVec& dvec, AlignedMapVec& vec) = 0;
};


#endif /* OPTIMIZER_H_ */
