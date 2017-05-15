#ifndef NETWORK_H_
#define NETWORK_H_

#include <Eigen/Core>
#include <vector>
#include <stdexcept>
#include "Config.h"
#include "Layer.h"
#include "Output.h"
#include "Callback.h"
#include "Utils/Random.h"

class Network
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::RowVectorXi IntegerVector;

    std::vector<Layer*> m_layers;
    Output*             m_output;
    RNGType             m_default_rng;      // Built-in RNG
    RNGType&            m_rng;              // Points to the RNG provided by the user,
                                            // otherwise points to m_default_rng
    Callback            m_default_callback; // Default callback function
    Callback*           m_callback;         // Points to user-provided callback function,
                                            // otherwise points to m_default_callback

    // Check dimensions of layers
    void check_unit_sizes()
    {
        const int nlayer = m_layers.size();
        if(nlayer <= 1)
            return;

        for(int i = 1; i < nlayer; i++)
        {
            if(m_layers[i]->in_size() != m_layers[i - 1]->out_size())
                throw std::invalid_argument("Unit sizes do not match");
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
            throw std::invalid_argument("Input data have incorrect dimension");
        m_layers[0]->forward(input);

        // The following layers
        for(int i = 1; i < nlayer; i++)
        {
            m_layers[i]->forward(m_layers[i - 1]->output());
        }
    }

    // Let each layer compute its gradients of the parameters
    // target has two versions: Matrix and RowVectorXi
    // The RowVectorXi version is used in classification problems where each
    // element is a class label
    template <typename TargetType>
    void backprop(const Matrix& input, const TargetType& target)
    {
        const int nlayer = m_layers.size();
        if(nlayer <= 0)
            return;

        Layer* first_layer = m_layers[0];
        Layer* last_layer = m_layers[nlayer - 1];

        // Let output layer compute back-propagation data
        m_output->evaluate(last_layer->output(), target);

        // If there is only one hidden layer, "prev_layer_data" will be the input data
        if(nlayer == 1)
        {
            first_layer->backprop(input, m_output->backprop_data());
            return;
        }

        // Compute gradients for the last hidden layer
        last_layer->backprop(m_layers[nlayer - 2]->output(), m_output->backprop_data());
        // Compute gradients for all the hidden layers except for the first one and the last one
        for(int i = nlayer - 2; i > 0; i--)
        {
            m_layers[i]->backprop(m_layers[i - 1]->output(), m_layers[i + 1]->backprop_data());
        }
        // Compute gradients for the first layer
        first_layer->backprop(input, m_layers[1]->backprop_data());
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
        m_rng(m_default_rng),
        m_default_callback(),
        m_callback(&m_default_callback)
    {}

    Network(RNGType& rng) :
        m_output(NULL),
        m_default_rng(1),
        m_rng(rng),
        m_default_callback(),
        m_callback(&m_default_callback)
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

    // NOTE: layer is a pointer that will be deleted by the network object, so don't delete it outside
    void add_layer(Layer* layer)
    {
        m_layers.push_back(layer);
    }

    // NOTE: output is a pointer that will be deleted by the network object, so don't delete it outside
    void add_output(Output* output)
    {
        if(m_output)
            delete m_output;

        m_output = output;
    }

    void set_callback(Callback& callback)
    {
        m_callback = &callback;
    }

    // Initialize parameters using N(mu, sigma^2) distribution
    // Random seed will be set if seed > 0
    void init(const Scalar& mu = Scalar(0), const Scalar& sigma = Scalar(0.01), int seed = -1)
    {
        check_unit_sizes();

        if(seed > 0)
            m_rng.seed(seed);

        const int nlayer = m_layers.size();
        for(int i = 0; i < nlayer; i++)
        {
            m_layers[i]->init(mu, sigma, m_rng);
        }
    }

    // Compute the current loss function value
    // This function is mainly used to report loss function value inside fit(),
    // and can be assumed to be called after backprop()
    template <typename TargetType>
    Scalar loss(const TargetType& target) const
    {
        const Layer* last_layer = m_layers.back();

        return m_output->loss(last_layer->output(), target);
    }

    std::vector< std::vector<Scalar> > get_parameters() const
    {
        const int nlayer = m_layers.size();
        std::vector< std::vector<Scalar> > res;
        res.reserve(nlayer);
        for(int i = 0; i < nlayer; i++)
        {
            res.push_back(m_layers[i]->get_parameters());
        }

        return res;
    }

    void set_parameters(const std::vector< std::vector<Scalar> >& param)
    {
        const int nlayer = m_layers.size();
        if(static_cast<int>(param.size()) != nlayer)
            throw std::invalid_argument("Parameter size does not match");

        for(int i = 0; i < nlayer; i++)
        {
            m_layers[i]->set_parameters(param[i]);
        }
    }

    std::vector< std::vector<Scalar> > get_derivatives() const
    {
        const int nlayer = m_layers.size();
        std::vector< std::vector<Scalar> > res;
        res.reserve(nlayer);
        for(int i = 0; i < nlayer; i++)
        {
            res.push_back(m_layers[i]->get_derivatives());
        }

        return res;
    }

    /*
    template <typename TargetType>
    void check_gradient(const Matrix& input, const TargetType& target)
    {
        this->forward(input);
        this->backprop(input, target);
        std::vector< std::vector<Scalar> > param = this->get_parameters();
        std::vector< std::vector<Scalar> > deriv = this->get_derivatives();

        const Scalar eps = 1e-4;
        for(unsigned int i = 0; i < deriv.size(); i++)
        {
            for(unsigned int j = 0; j < deriv[i].size(); j++)
            {
                Scalar old = param[i][j];

                param[i][j] -= eps;
                this->set_parameters(param);
                this->forward(input);
                this->backprop(input, target);
                Scalar loss_pre = this->loss(target);

                param[i][j] += eps * 2;
                this->set_parameters(param);
                this->forward(input);
                this->backprop(input, target);
                Scalar loss_post = this->loss(target);

                Scalar dest = (loss_post - loss_pre) / eps / 2;

                std::cout << "i = " << i << ", j = " << j <<
                ", d = " << deriv[i][j] << ", dest = " << (loss_post - loss_pre) / eps / 2 <<
                ", diff = " << dest - deriv[i][j] << std::endl;

                param[i][j] = old;
            }
        }
        this->set_parameters(param);
    }
    */

    // Fit a model
    // Random seed will be set if seed > 0
    template <typename DerivedX, typename DerivedY>
    bool fit(Optimizer& opt, const Eigen::MatrixBase<DerivedX>& x, const Eigen::MatrixBase<DerivedY>& y,
             int batch_size, int epoch, int seed = -1)
    {
        // We do not directly use PlainObjectX since it may be row-majored if x is passed as mat.transpose()
        // We want to force XType and YType to be column-majored
        typedef typename Eigen::MatrixBase<DerivedX>::PlainObject PlainObjectX;
        typedef typename Eigen::MatrixBase<DerivedX>::PlainObject PlainObjectY;
        typedef Eigen::Matrix<typename PlainObjectX::Scalar, PlainObjectX::RowsAtCompileTime, PlainObjectX::ColsAtCompileTime> XType;
        typedef Eigen::Matrix<typename PlainObjectY::Scalar, PlainObjectY::RowsAtCompileTime, PlainObjectY::ColsAtCompileTime> YType;

        const int nlayer = m_layers.size();
        if(nlayer <= 0)
            return false;

        // Reset optimizer
        opt.reset();

        // Create shuffled mini-batches
        if(seed > 0)
            m_rng.seed(seed);

        std::vector<XType> x_batches;
        std::vector<YType> y_batches;
        const int nbatch = create_shuffled_batches(x, y, batch_size, m_rng, x_batches, y_batches);

        // Set up callback parameters
        m_callback->m_nbatch = nbatch;
        m_callback->m_nepoch = epoch;

        // Iterations on the whole data set
        for(int k = 0; k < epoch; k++)
        {
            m_callback->m_epoch_id = k;

            // Train on each mini-batch
            for(int i = 0; i < nbatch; i++)
            {
                m_callback->m_batch_id = i;
                m_callback->pre_training_batch(this, x_batches[i], y_batches[i]);

                this->forward(x_batches[i]);
                this->backprop(x_batches[i], y_batches[i]);
                this->update(opt);

                m_callback->post_training_batch(this, x_batches[i], y_batches[i]);
            }
        }

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
