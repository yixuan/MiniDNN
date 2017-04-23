#ifndef LAYER_FULLYCONNECTED_H_
#define LAYER_FULLYCONNECTED_H_

#include <Eigen/Core>
#include "../Config.h"
#include "../Layer.h"

template <typename Activation>
class FullyConnected: public Layer
{
private:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

	Matrix m_weight;  // Weight parameters, insize x outsize
	Vector m_bias;    // Bias parameters, outsize x 1
	Matrix m_dw;      // Derivative of weights
	Vector m_db;      // Derivative of bias
	Matrix m_a;       // Output of this layer, a = act(z), z = W' * in + b
	Matrix m_bpdata;  // Data for back-propagation

public:
	FullyConnected(const int insize, const int outsize) :
		Layer(insize, outsize)
	{}

	void init()
	{
		m_weight.resize(this->m_insize, this->m_outsize);
		m_bias.resize(this->m_outsize);
		m_dw.resize(this->m_insize, this->m_outsize);
		m_db.resize(this->m_outsize);

		// Set random coefficients
		m_weight.setRandom();
		m_bias.setRandom();
	}

	void forward(const Matrix& prev_layer_data)
	{
		const int nobs = prev_layer_data.cols();
		// Use m_bpdata to temporarily store linear term z = W' * in + b
		m_bpdata.resize(this->m_outsize, nobs);
		m_bpdata.noalias() = m_weight.transpose() * prev_layer_data;
		m_bpdata.colwise() += m_bias;

		// Apply activation function
		m_a.resize(this->m_outsize, nobs);
		Activation::activate(m_bpdata, m_a);
	}

	const Matrix& output()
	{
		return m_a;
	}

	// prev: insize x nobs
	// next: outsize x nobs
	void backprop(const Matrix& prev_layer_data, const Matrix& next_layer_data)
	{
		const int nobs = prev_layer_data.cols();

		// After forward stage, m_bpdata contains z = W' * in + b
		// Now compute act'(z)
		Activation::deriv_activate(m_bpdata, m_a, m_bpdata);

		// Compute delta = bpdata .* act'(z)
		m_bpdata.array() *= next_layer_data.array();

		// Derivative for weights
		m_dw.noalias() = prev_layer_data * m_bpdata.transpose() / nobs;

		// Derivative for bias
		m_db.noalias() = m_bpdata.rowwise().mean();

		// Back-propagation data for previous layer
		m_bpdata = m_weight * m_bpdata;
	}

	const Matrix& backprop_data()
	{
		return m_bpdata;
	}

	void update(Optimizer& opt)
	{
		opt.update_mat(m_dw, m_weight);
		opt.update_vec(m_db, m_bias);
	}

	Vector parameters()
	{
		Vector res(m_weight.size() + m_bias.size());
		// Copy the data of weights and bias to a long vector
		std::copy(m_weight.data(), m_weight.data() + m_weight.size(), res.data());
		std::copy(m_bias.data(), m_bias.data() + m_bias.size(), res.data() + m_weight.size());

		return res;
	}

	Vector derivatives()
	{
		Vector res(m_dw.size() + m_db.size());
		// Copy the data of weights and bias to a long vector
		std::copy(m_dw.data(), m_dw.data() + m_dw.size(), res.data());
		std::copy(m_db.data(), m_db.data() + m_db.size(), res.data() + m_dw.size());

		return res;
	}
};



#endif /* LAYER_FULLYCONNECTED_H_ */
