#ifndef OUTPUT_H_
#define OUTPUT_H_

#include <Eigen/Core>
#include "Config.h"

// Output layer is a special layer that associates the last hidden layer with the target response variable
class Output
{
protected:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

public:
	virtual ~Output() {}

	// A combination of the forward stage and the back-propagation stage for the output layer
	// The computed derivative of the input should be stored in this layer, and can be retrieved by
	// the backprop_data() function
	virtual void evaluate(const Matrix& prev_layer_data, const Matrix& target) = 0;

	// The derivative of the input of this layer, which is also the derivative
	// of the output of previous layer
	virtual const Matrix& backprop_data() const = 0;

	// Compute the loss function value
	// This function can be assumed to be called after evaluate(), so that it can make use of the
	// intermediate result to save some computation
	virtual Scalar loss(const Matrix& prev_layer_data, const Matrix& target) const = 0;
};


#endif /* OUTPUT_H_ */
