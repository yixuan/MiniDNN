#ifndef OPTIMIZER_SGD_H_
#define OPTIMIZER_SGD_H_

class SGD: public Optimizer
{
private:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
	typedef Eigen::Array<Scalar, Eigen::Dynamic, 1> Array;

public:
	Scalar m_lrate;
	Scalar m_decay;

	SGD() :
		m_lrate(Scalar(0.01)), m_decay(Scalar(0))
	{}

	void reset()
	{
		m_lrate = Scalar(0.01);
		m_decay = Scalar(0);
	}

	void update_vec(const Vector& dvec, Vector& vec)
	{
		vec.noalias() -= m_lrate * (dvec + m_decay * vec);
	}

	void update_mat(const Matrix& dmat, Matrix& mat)
	{
		mat.noalias() -= m_lrate * (dmat + m_decay * mat);
	}
};


#endif /* OPTIMIZER_SGD_H_ */
