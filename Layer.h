#ifndef LAYER_H_
#define LAYER_H_

#include <Eigen/Core>
#include "Config.h"
#include "Optimizer.h"

class Layer
{
protected:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

	const int m_insize;  // Size of input units
	const int m_outsize; // Size of output units

public:
	Layer(const int insize, const int outsize) :
		m_insize(insize), m_outsize(outsize)
	{}

	int in_size() { return m_insize; }
	int out_size() { return m_outsize; }

	virtual ~Layer() {}

	// Initialize parameters
	virtual void init(const Scalar& mu, const Scalar& sigma, RNGType& rng) = 0;

	// Compute output given input from previous layer
	// Each column of prev_layer_data is an observation
	// The computed data should be stored in the layer, and can be retrieved by the output() function
	virtual void forward(const Matrix& prev_layer_data) = 0;

	// Get a constant reference to the output of this layer, after calling forward()
	// Each column is an observation
	virtual const Matrix& output() = 0;

	// Compute gradients using back-propagation
	// prev_layer_data contains the output of previous layer (which is also the input of this layer)
	// next_layer_data contains the back-propagation data from next layer
	virtual void backprop(const Matrix& prev_layer_data, const Matrix& next_layer_data) = 0;

	// Data for previous layer in back-propagation
	// Provides "next_layer_data" for the previous layer
	virtual const Matrix& backprop_data() = 0;

	// Update parameters given gradients
	virtual void update(Optimizer& opt) = 0;

	// Get serialized parameters
	virtual Vector parameters() = 0;

	// Get serialized gradients of parameters
	virtual Vector derivatives() = 0;
};


#endif /* LAYER_H_ */
