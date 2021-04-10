#ifndef MINIDNN_NETWORK_H_
#define MINIDNN_NETWORK_H_

#include <Eigen/Core>
#include <vector>
#include <memory>
#include <utility>
#include <map>
#include <stdexcept>
#include "Config.h"
#include "Layer.h"
#include "Output.h"
#include "Callback.h"
#include "Utils/IO.h"
#include "Utils/Factory.h"

namespace MiniDNN
{


///
/// \defgroup Network Neural Network Model
///

///
/// \ingroup Network
///
/// This class represents a neural network model that typically consists of a
/// number of hidden layers and an output layer. It provides functions for
/// network building, model fitting, and prediction, etc.
///
class Network
{
private:
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using IntegerVector = Eigen::RowVectorXi;
    using MetaInfo = std::map<std::string, int>;

    RNG&                                m_rng;      // Reference to the RNG provided by the user.
    std::vector<std::unique_ptr<Layer>> m_layers;   // Pointers to hidden layers
    std::unique_ptr<Output>             m_output;   // The output layer
    std::unique_ptr<Callback>           m_callback; // Points to user-provided callback function
                                                    // Default is the silent one

    // Check dimensions of layers
    void check_unit_sizes() const
    {
        const int nlayer = num_layers();
        if (nlayer <= 1)
            return;

        int prev_out_size = m_layers[0]->out_size();
        for (int i = 1; i < nlayer; i++)
        {
            // For activation layers, in_size and out_size are both zero.
            // We simply ignore them
            if (m_layers[i]->in_size() == 0)
            {
                continue;
            }
            else if (m_layers[i]->in_size() != prev_out_size)
            {
                throw std::invalid_argument("[class Network]: Unit sizes do not match");
            }
            prev_out_size = m_layers[i]->out_size();
        }
    }

    // Let each layer compute its output
    void forward(const Matrix& input)
    {
        const int nlayer = num_layers();
        if (nlayer <= 0)
            return;

        // First layer
        if (input.rows() != m_layers[0]->in_size())
        {
            throw std::invalid_argument("[class Network]: Input data have incorrect dimension");
        }
        m_layers[0]->forward(input);

        // The following layers
        for (int i = 1; i < nlayer; i++)
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
        const int nlayer = num_layers();
        if (nlayer <= 0)
            return;

        using LayerPtr = std::unique_ptr<Layer>;
        LayerPtr first_layer = m_layers[0];
        LayerPtr last_layer = m_layers[nlayer - 1];
        // Let output layer compute back-propagation data
        m_output->check_target_data(target);
        m_output->evaluate(last_layer->output(), target);

        // If there is only one hidden layer, "prev_layer_data" will be the input data
        if (nlayer == 1)
        {
            first_layer->backprop(input, m_output->backprop_data());
            return;
        }

        // Compute gradients for the last hidden layer
        last_layer->backprop(m_layers[nlayer - 2]->output(), m_output->backprop_data());

        // Compute gradients for all the hidden layers except for the first one and the last one
        for (int i = nlayer - 2; i > 0; i--)
        {
            m_layers[i]->backprop(m_layers[i - 1]->output(), m_layers[i + 1]->backprop_data());
        }

        // Compute gradients for the first layer
        first_layer->backprop(input, m_layers[1]->backprop_data());
    }

    // Update parameters
    void update(Optimizer& opt)
    {
        const int nlayer = num_layers();
        if (nlayer <= 0)
            return;

        for (int i = 0; i < nlayer; i++)
        {
            m_layers[i]->update(opt);
        }
    }

    // Get the meta information of the network, used to export the NN model
    MetaInfo get_meta_info() const
    {
        const int nlayer = num_layers();
        MetaInfo map;
        map.insert(std::make_pair("Nlayers", nlayer));

        for (int i = 0; i < nlayer; i++)
        {
            m_layers[i]->fill_meta_info(map, i);
        }

        map.insert(std::make_pair("OutputLayer", internal::output_id(m_output->output_type())));
        return map;
    }

public:
    ///
    /// Constructor that creates an empty neural network.
    ///
    /// \param rng A user-provided random number generator.
    ///
    Network(RNG& rng) :
        m_rng(rng),
        m_output(nullptr),
        m_callback(new Callback())
    {}

    ///
    /// Add a hidden layer to the neural network.
    ///
    /// \param layer A Layer object, typically constructed from
    ///              layer classes such as FullyConnected and Convolutional.
    ///              If layer is an lvalue, it will be copied; if it is an rvalue,
    ///              it will be moved into the network object.
    ///
    template <T>
    void add_layer(T&& layer)
    {
        using LayerType = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
        m_layers.emplace_back(new LayerType(std::forward<T>(layer)));
    }

    ///
    /// Set the output layer of the neural network.
    ///
    /// \param output An Output object, typically constructed from
    ///               output layer classes such as RegressionMSE and MultiClassEntropy.
    ///               If output is an lvalue, it will be copied; if it is an rvalue,
    ///               it will be moved into the network object.
    ///
    template <T>
    void set_output(T&& output)
    {
        using OutputType = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
        m_output.set(new OutputType(std::forward<T>(output)));
    }

    ///
    /// Number of hidden layers in the network.
    ///
    int num_layers() const
    {
        return m_layers.size();
    }

    ///
    /// Get the list of hidden layers of the network.
    ///
    std::vector<const Layer*> get_layers() const
    {
        const int nlayer = num_layers();
        std::vector<const Layer*> layers(nlayer);
        for (int i = 0; i < nlayer; i++)
            layers[i] = m_layers[i].get();
        return layers;
    }

    ///
    /// Get the output layer.
    ///
    const Output& get_output() const
    {
        return *m_output;
    }

    ///
    /// Set the callback function that can be called during model fitting.
    ///
    /// \param callback A user-provided callback function object that inherits
    ///                 from the default Callback class.
    ///                 If callback is an lvalue, it will be copied; if it is an rvalue,
    ///                 it will be moved into the network object.
    ///
    template <T>
    void set_callback(T&& callback)
    {
        using CallBackType = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
        m_callback.reset(new CallBackType(std::forward<T>(callback)));
    }

    ///
    /// Initialize layer parameters in the network.
    ///
    /// \param initializer The initializer. See the Initializer class.
    ///
    void init(const Initializer& initializer)
    {
        check_unit_sizes();
        const int nlayer = num_layers();
        for (int i = 0; i < nlayer; i++)
        {
            m_layers[i]->init(initializer, m_rng);
        }
    }

    ///
    /// Get the serialized layer parameters.
    ///
    std::vector<std::vector<Scalar>> get_parameters() const
    {
        const int nlayer = num_layers();
        std::vector<std::vector<Scalar>> res;
        res.reserve(nlayer);

        for (int i = 0; i < nlayer; i++)
        {
            res.push_back(m_layers[i]->get_parameters());
        }

        return res;
    }

    ///
    /// Set the layer parameters.
    ///
    /// \param param Serialized layer parameters.
    ///
    void set_parameters(const std::vector<std::vector<Scalar>>& param)
    {
        const int nlayer = num_layers();
        if (static_cast<int>(param.size()) != nlayer)
        {
            throw std::invalid_argument("[class Network]: Parameter size does not match");
        }

        for (int i = 0; i < nlayer; i++)
        {
            m_layers[i]->set_parameters(param[i]);
        }
    }

    ///
    /// Get the serialized derivatives of layer parameters.
    ///
    std::vector<std::vector<Scalar>> get_derivatives() const
    {
        const int nlayer = num_layers();
        std::vector<std::vector<Scalar>> res;
        res.reserve(nlayer);

        for (int i = 0; i < nlayer; i++)
        {
            res.push_back(m_layers[i]->get_derivatives());
        }

        return res;
    }

    ///
    /// Fit the model based on the given data.
    ///
    /// \param opt        An object that inherits from the Optimizer class, indicating the optimization algorithm to use.
    /// \param x          The predictors. Each column is an observation.
    /// \param y          The response variable. Each column is an observation.
    /// \param batch_size Mini-batch size.
    /// \param epoch      Number of epochs of training.
    /// \param seed       Set the random seed of the %RNG if `seed > 0`, otherwise
    ///                   use the current random state.
    ///
    template <typename DerivedX, typename DerivedY>
    bool fit(Optimizer& opt, const Eigen::MatrixBase<DerivedX>& x, const Eigen::MatrixBase<DerivedY>& y,
             int batch_size, int epoch)
    {
        // We do not directly use PlainObjectX since it may be row-majored if x is passed as mat.transpose()
        // We want to force XType and YType to be column-majored
        using PlainObjectX = typename Eigen::MatrixBase<DerivedX>::PlainObject;
        using PlainObjectY = typename Eigen::MatrixBase<DerivedY>::PlainObject;
        using XType = Eigen::Matrix<typename PlainObjectX::Scalar, PlainObjectX::RowsAtCompileTime, PlainObjectX::ColsAtCompileTime>;
        using YType = Eigen::Matrix<typename PlainObjectY::Scalar, PlainObjectY::RowsAtCompileTime, PlainObjectY::ColsAtCompileTime>;

        const int nlayer = num_layers();
        if (nlayer <= 0)
            return false;

        // Reset optimizer
        opt.reset();

        std::vector<XType> x_batches;
        std::vector<YType> y_batches;
        const int nbatch = internal::create_shuffled_batches(x, y, batch_size, m_rng, x_batches, y_batches);
        // Set up callback parameters
        m_callback->m_nbatch = nbatch;
        m_callback->m_nepoch = epoch;

        // Iterations on the whole data set
        for (int k = 0; k < epoch; k++)
        {
            m_callback->m_epoch_id = k;

            // Train on each mini-batch
            for (int i = 0; i < nbatch; i++)
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

    ///
    /// Use the fitted model to make predictions.
    ///
    /// \param x The predictors. Each column is an observation.
    ///
    Matrix predict(const Matrix& x)
    {
        const int nlayer = num_layers();
        if (nlayer <= 0)
            return Matrix();

        this->forward(x);
        return m_layers[nlayer - 1]->output();
    }

    ///
    /// Export the network to files.
    ///
    /// \param folder   The folder where the network is saved.
    /// \param filename The filename for the network.
    ///
    void export_net(const std::string& folder, const std::string& filename) const
    {
        bool created = internal::create_directory(folder);
        if (!created)
            throw std::runtime_error("[class Network]: Folder creation failed");

        MetaInfo map = this->get_meta_info();
        internal::write_map(folder + "/" + filename, map);
        std::vector<std::vector<Scalar>> params = this->get_parameters();
        internal::write_parameters(folder, filename, params);
    }

    ///
    /// Read in a network from files.
    ///
    /// \param folder   The folder where the network is saved.
    /// \param filename The filename for the network.
    ///
    void read_net(const std::string& folder, const std::string& filename)
    {
        MetaInfo map;
        internal::read_map(folder + "/" + filename, map);
        int nlayer = map.find("Nlayers")->second;
        std::vector<std::vector<Scalar>> params = internal::read_parameters(folder, filename, nlayer);
        m_layers.clear();

        for (int i = 0; i < nlayer; i++)
        {
            this->add_layer(internal::create_layer(map, i));
        }

        this->set_parameters(params);
        this->set_output(internal::create_output(map));
    }
};


} // namespace MiniDNN


#endif // MINIDNN_NETWORK_H_
