#ifndef LAYER_H_
#define LAYER_H_

#include <Eigen/Core>
#include <vector>
#include "Config.h"
#include "Optimizer.h"

class Layer
{
protected:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

    const int m_in_size;  // Size of input units
    const int m_out_size; // Size of output units

public:
    Layer(const int in_size, const int out_size) :
        m_in_size(in_size), m_out_size(out_size)
    {}

    virtual ~Layer() {}

    int in_size() const { return m_in_size; }
    int out_size() const { return m_out_size; }

    // Initialize parameters using N(mu, sigma^2) distribution
    virtual void init(const Scalar& mu, const Scalar& sigma, RNGType& rng) = 0;

    // Compute the output of this layer
    // prev_layer_data is the output of previous layer, which is also the input of this layer
    // Each column of prev_layer_data is an observation
    // The computed data should be stored in the layer, and can be retrieved by the output() function
    virtual void forward(const Matrix& prev_layer_data) = 0;

    // Get a constant reference to the output of this layer, after calling forward()
    // Each column is an observation
    virtual const Matrix& output() const = 0;

    // Compute gradients using back-propagation
    // prev_layer_data is the output of previous layer, which is also the input of this layer
    // next_layer_data is the derivative of the input of next layer, which is also the derivative
    // of the output of this layer
    virtual void backprop(const Matrix& prev_layer_data, const Matrix& next_layer_data) = 0;

    // The derivative of the input of this layer, which is also the derivative
    // of the output of previous layer
    virtual const Matrix& backprop_data() const = 0;

    // Update parameters given gradients
    virtual void update(Optimizer& opt) = 0;

    // Get serialized parameters
    virtual std::vector<Scalar> parameters() const = 0;

    // Get serialized gradients of parameters
    virtual std::vector<Scalar> derivatives() const = 0;
};


#endif /* LAYER_H_ */
