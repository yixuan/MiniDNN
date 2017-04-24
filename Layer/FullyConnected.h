#ifndef LAYER_FULLYCONNECTED_H_
#define LAYER_FULLYCONNECTED_H_

#include <Eigen/Core>
#include "../Config.h"
#include "../Layer.h"
#include "../Utils/Random.h"

template <typename Activation>
class FullyConnected: public Layer
{
private:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
	typedef Vector::ConstAlignedMapType ConstAlignedMapVec;
	typedef Vector::AlignedMapType AlignedMapVec;

	Matrix m_weight;  // Weight parameters, W(insize x outsize)
	Vector m_bias;    // Bias parameters, b(outsize x 1)
	Matrix m_dw;      // Derivative of weights
	Vector m_db;      // Derivative of bias
	Matrix m_a;       // Output of this layer, a = act(z), z = W' * in + b
	Matrix m_din;     // Derivative of the input of this layer.
	                  // Note that input of this layer is also the output of previous layer

public:
	FullyConnected(const int insize, const int outsize) :
		Layer(insize, outsize)
	{}

	void init(const Scalar& mu, const Scalar& sigma, RNGType& rng)
	{
		m_weight.resize(this->m_insize, this->m_outsize);
		m_bias.resize(this->m_outsize);
		m_dw.resize(this->m_insize, this->m_outsize);
		m_db.resize(this->m_outsize);

		// Set random coefficients
		set_normal_random(m_weight.data(), m_weight.size(), rng, mu, sigma);
		set_normal_random(m_bias.data(), m_bias.size(), rng, mu, sigma);
	}

	void forward(const Matrix& prev_layer_data)
	{
		const int nobs = prev_layer_data.cols();
		// Use m_din to temporarily store linear term z = W' * in + b
		m_din.resize(this->m_outsize, nobs);
		m_din.noalias() = m_weight.transpose() * prev_layer_data;
		m_din.colwise() += m_bias;

		// Apply activation function
		m_a.resize(this->m_outsize, nobs);
		Activation::activate(m_din, m_a);
	}

	const Matrix& output() const
	{
		return m_a;
	}

	// prev: insize x nobs
	// next: outsize x nobs
	void backprop(const Matrix& prev_layer_data, const Matrix& next_layer_data)
	{
		const int nobs = prev_layer_data.cols();

		// After forward stage, m_din contains z = W' * in + b
		// Now we need to calculate d_L / d_z = (d_a / d_z) * (d_L / d_a)
		// d_L / d_a is computed in the next layer, contained in next_layer_data
		// The Jacobian matrix J = d_a / d_z is determined by the activation function
		Activation::apply_jacobian(m_din, m_a, next_layer_data, m_din);

		// Derivative for weights, d_L / d_W = (d_L / d_z) * in'
		m_dw.noalias() = prev_layer_data * m_din.transpose() / nobs;

		// Derivative for bias, d_L / d_b = d_L / d_z
		m_db.noalias() = m_din.rowwise().mean();

		// Compute d_L / d_in = W * (d_L / d_z)
		m_din = m_weight * m_din;
	}

	const Matrix& backprop_data() const
	{
		return m_din;
	}

	void update(Optimizer& opt)
	{
		ConstAlignedMapVec dw(m_dw.data(), m_dw.size());
		ConstAlignedMapVec db(m_db.data(), m_db.size());
		AlignedMapVec      w(m_weight.data(), m_weight.size());
		AlignedMapVec      b(m_bias.data(), m_bias.size());

		opt.update(dw, w);
		opt.update(db, b);
	}

	std::vector<Scalar> parameters() const
	{
		std::vector<Scalar> res(m_weight.size() + m_bias.size());
		// Copy the data of weights and bias to a long vector
		std::copy(m_weight.data(), m_weight.data() + m_weight.size(), res.begin());
		std::copy(m_bias.data(), m_bias.data() + m_bias.size(), res.begin() + m_weight.size());

		return res;
	}

	std::vector<Scalar> derivatives() const
	{
		std::vector<Scalar> res(m_dw.size() + m_db.size());
		// Copy the data of weights and bias to a long vector
		std::copy(m_dw.data(), m_dw.data() + m_dw.size(), res.begin());
		std::copy(m_db.data(), m_db.data() + m_db.size(), res.begin() + m_dw.size());

		return res;
	}
};



#endif /* LAYER_FULLYCONNECTED_H_ */
