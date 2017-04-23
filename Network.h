#ifndef NETWORK_H_
#define NETWORK_H_

#include <Eigen/Core>
#include <memory>
#include <vector>
#include <exception>
#include "Config.h"
#include "Layer.h"
#include "Output.h"
#include "Utils/Random.h"

class Network
{
private:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

	std::vector<Layer*> m_layers;
	Output*             m_output;
	RNGType             m_default_rng;  // Built-in RNG
	RNGType&            m_rng;          // Points to the RNG provided by the user,
	                                    // otherwise points to m_default_rng

	// Check dimensions of layers
	void check_unit_sizes()
	{
		const int nlayer = m_layers.size();
		if(nlayer <= 1)
			return;

		for(int i = 1; i < nlayer; i++)
		{
			if(m_layers[i]->in_size() != m_layers[i - 1]->out_size())
				throw std::domain_error("Unit sizes do not match");
		}
	}

	// Let each layer compute its output
	void forward(const Matrix& input)
	{
		const int nlayer = m_layers.size();
		if(nlayer <= 0)
			return;

		// First layer
		if(input.rows() != m_layers[0]->in_size())
			throw std::domain_error("Input data have incorrect dimension");
		m_layers[0]->forward(input);

		// The following layers
		for(int i = 1; i < nlayer; i++)
		{
			m_layers[i]->forward(m_layers[i - 1]->output());
		}
	}

	// Let each layer compute its gradients of the parameters
	// This function returns the current loss function value if compute_loss is true
	Scalar backprop(const Matrix& input, const Matrix& target, bool compute_loss = false)
	{
		const int nlayer = m_layers.size();
		if(nlayer <= 0)
			return Scalar(0);

		Layer* first_layer = m_layers[0];
		Layer* last_layer = m_layers[nlayer - 1];

		// Let output compute back-propagation data
		Scalar loss = m_output->evaluate(last_layer->output(), target, compute_loss);

		// If there is only one layer, "prev_layer_data" will be the input data
		if(nlayer == 1)
		{
			first_layer->backprop(input, m_output->backprop_data());
			return loss;
		}

		// Compute gradients for last layer
		last_layer->backprop(m_layers[nlayer - 2]->output(), m_output->backprop_data());
		// Compute gradients for all the layers except for the first one and the last one
		for(int i = nlayer - 2; i > 0; i--)
		{
			m_layers[i]->backprop(m_layers[i - 1]->output(), m_layers[i + 1]->backprop_data());
		}
		// Compute gradients for first layer
		first_layer->backprop(input, m_layers[1]->backprop_data());

		return loss;
	}

	// Update parameters
	void update(Optimizer& opt)
	{
		const int nlayer = m_layers.size();
		if(nlayer <= 0)
			return;

		for(int i = 0; i < nlayer; i++)
		{
			m_layers[i]->update(opt);
		}
	}

public:
	Network() :
		m_output(NULL),
		m_default_rng(1),
		m_rng(m_default_rng)
	{}

	Network(RNGType& rng) :
		m_output(NULL),
		m_default_rng(1),
		m_rng(rng)
    {}

	~Network()
	{
		const int nlayer = m_layers.size();
		for(int i = 0; i < nlayer; i++)
		{
			delete m_layers[i];
		}

		if(m_output)
			delete m_output;
	}

	void add_layer(Layer* layer)
	{
		m_layers.push_back(layer);
	}

	void add_output(Output* output)
	{
		if(m_output)
			delete m_output;

		m_output = output;
	}

	void init(Scalar sd = Scalar(0.01), int seed = -1)
	{
		check_unit_sizes();

		const int nlayer = m_layers.size();
		for(int i = 0; i < nlayer; i++)
		{
			m_layers[i]->init();
		}
	}

	// Fit a model
	bool fit(Optimizer& opt, const Matrix& x, const Matrix& y, int batch_size, int epoch,
			 int seed = -1)
	{
		const int nlayer = m_layers.size();
		if(nlayer <= 0)
			return false;

		const int nobs = x.cols();
		const int dimx = x.rows();
		const int dimy = y.rows();

		// Randomly shuffle the IDs
		if(seed > 0)
			m_rng.seed(seed);
		Eigen::VectorXi id = Eigen::VectorXi::LinSpaced(nobs, 0, nobs - 1);
		shuffle(id.data(), id.size(), m_rng);

		// Compute batch size
		if(batch_size > nobs)
			batch_size = nobs;
		const int nbatch = (nobs - 1) / batch_size + 1;
		const int last_batch_size = nobs - (nbatch - 1) * batch_size;

		// Create shuffled data
		Matrix* batchx = new Matrix[nbatch];
		Matrix* batchy = new Matrix[nbatch];
		for(int i = 0; i < nbatch; i++)
		{
			const int bsize = (i == nbatch - 1) ? last_batch_size : batch_size;
			batchx[i].resize(dimx, bsize);
			batchy[i].resize(dimy, bsize);
			// Copy data
			const int offset = i * batch_size;
			for(int j = 0; j < bsize; j++)
			{
				batchx[i].col(j).noalias() = x.col(id[offset + j]);
				batchy[i].col(j).noalias() = y.col(id[offset + j]);
			}
		}

		// Iterations on the whole data set
		for(int k = 0; k < epoch; k++)
		{
			// Train on each mini-batch
			for(int i = 0; i < nbatch; i++)
			{
				this->forward(batchx[i]);
				Scalar loss = this->backprop(batchx[i], batchy[i], true);
				std::cout << "loss = " << loss << std::endl;
				this->update(opt);
			}
		}

		delete [] batchx;
		delete [] batchy;

		return true;
	}

	// Make predictions
	Matrix predict(const Matrix& x)
	{
		const int nlayer = m_layers.size();
		if(nlayer <= 0)
			return Matrix();
		
		this->forward(x);
		return m_layers[nlayer - 1]->output();
	}
};


#endif /* NETWORK_H_ */
