#ifndef MINIDNN_ACTIVATION_H_
#define MINIDNN_ACTIVATION_H_

#include <Eigen/Core>
#include "Layer.h"

namespace MiniDNN
{


///
/// \ingroup Layers
///
/// The interface of activation layers.
///
class Activation: public Layer
{
protected:
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using MetaInfo = std::map<std::string, int>;

    Matrix m_out;  // Output of the activation layer, a = f(z)
    Matrix m_din;  // Derivative of the input, dl/dz

public:
    ///
    /// Constructor.
    ///
    ///
    // Typically we do not care about the input and output sizes of activation
    // layers, so we set them to zero
    Activation() :
        Layer(0, 0)
    {}

    ///
    /// Virtual destructor.
    ///
    virtual ~Activation() {}

    // For most activation layers, there are no parameters to initialize
    void init(const Initializer& initializer, RNG& rng) override {}
    void init() override {};

    // Derived class needs to compute a = f(z)
    // z => prev_layer_data [d x n]
    // a => m_out [d x n]
    // virtual void forward(const Matrix& prev_layer_data) = 0;
    const Matrix& output() const override { return m_out; }

    // Derived class needs to compute dl/dz
    // z     => prev_layer_data [d x n]
    // dl/da => next_layer_data [d x n]
    // dl/dz => m_din [d x n]
    // virtual void backprop(const Matrix& prev_layer_data, const Matrix& next_layer_data) = 0;
    const Matrix& backprop_data() const override { return m_din; }

    // For most activation layers, there are no parameters
    void update(Optimizer& opt) override {};
    std::vector<Scalar> get_parameters() const override
    {
        return std::vector<Scalar>();
    }
    std::vector<Scalar> get_derivatives() const override
    {
        return std::vector<Scalar>();
    }
};


} // namespace MiniDNN


#endif // MINIDNN_ACTIVATION_H_
