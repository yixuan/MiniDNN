#ifndef OUTPUT_REGRESSIONMSE_H_
#define OUTPUT_REGRESSIONMSE_H_

#include <Eigen/Core>
#include <exception>
#include "../Config.h"

class RegressionMSE: public Output
{
private:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

	Matrix m_bpdata;  // Data to pass to previous layer in back-propagation

public:
	Scalar evaluate(const Matrix& layer_output, const Matrix& target, bool compute_loss = false)
	{
		// Check dimension
		const int nobs = layer_output.cols();
		if(target.cols() != nobs)
			throw std::domain_error("Target data have incorrect dimension");

		// Compute data for back propagation
		m_bpdata.resize(layer_output.rows(), nobs);
		m_bpdata.noalias() = layer_output - target;

		Scalar loss = Scalar(0);
		// 0.5 * ||yhat - y||^2
		if(compute_loss)
		{
			loss = m_bpdata.squaredNorm() / nobs * Scalar(0.5);
		}

		return loss;
	}

	const Matrix& backprop_data()
	{
		return m_bpdata;
	}
};


#endif /* OUTPUT_REGRESSIONMSE_H_ */
