#ifndef OPTIMIZER_ADAGRAD_H_
#define OPTIMIZER_ADAGRAD_H_

#include <Eigen/Core>
#include "../Config.h"
#include "../Optimizer.h"
#include "../Utils/sparsepp.h"

class Adagrad: public Optimizer
{
private:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
	typedef Eigen::Array<Scalar, Eigen::Dynamic, 1> Array;
	typedef Vector::ConstAlignedMapType ConstAlignedMapVec;
	typedef Vector::AlignedMapType AlignedMapVec;

	spp::sparse_hash_map<const Scalar*, Array> m_history;

public:
	Scalar m_lrate;
	Scalar m_eps;

	Adagrad() :
		m_lrate(Scalar(0.01)), m_eps(Scalar(1e-8))
	{}

	void reset()
	{
		m_lrate = Scalar(0.01);
		m_eps = Scalar(1e-8);
		m_history.clear();
	}

	void update(ConstAlignedMapVec& dvec, AlignedMapVec& vec)
	{
		// Get the G vector associated with this gradient
		Array& grad_square = m_history[dvec.data()];
		// If length is zero, initialize it
		if(grad_square.size() == 0)
		{
			grad_square.resize(dvec.size());
			grad_square.setZero();
		}
		// Update G vector
		grad_square += dvec.array().square();
		// Update parameters
		vec.array() -= m_lrate * dvec.array() / (grad_square + m_eps).sqrt();
	}
};


#endif /* OPTIMIZER_ADAGRAD_H_ */
