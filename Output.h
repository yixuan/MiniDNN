#ifndef OUTPUT_H_
#define OUTPUT_H_

#include <Eigen/Core>
#include "Config.h"

class Output
{
protected:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

public:
	virtual ~Output() {}

	// Compute back-propagation data that can be retrieved by the backprop_data() function
	// Optionally compute loss function value
	virtual Scalar evaluate(const Matrix& layer_output, const Matrix& target, bool compute_loss = false) = 0;

	// Data for previous layer in back-propagation
	virtual const Matrix& backprop_data() = 0;
};


#endif /* OUTPUT_H_ */
