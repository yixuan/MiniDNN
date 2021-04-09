#ifndef MINIDNN_OPTIMIZER_RMSPROP_H_
#define MINIDNN_OPTIMIZER_RMSPROP_H_

#include <Eigen/Core>
#include <unordered_map>
#include "../Config.h"
#include "../Optimizer.h"

namespace MiniDNN
{


///
/// \ingroup Optimizers
///
/// The RMSProp algorithm.
///
class RMSProp: public Optimizer
{
private:
    using Array = Eigen::Array<Scalar, Eigen::Dynamic, 1>;
    using Optimizer::Vector;
    using Optimizer::ConstAlignedMapVec;
    using Optimizer::AlignedMapVec;

    std::unordered_map<const Scalar*, Array> m_history;

public:
    Scalar m_lrate;
    Scalar m_eps;
    Scalar m_gamma;

    RMSProp(Scalar lrate = Scalar(0.001), Scalar eps = Scalar(1e-6),
            Scalar gamma = Scalar(0.9)) :
        m_lrate(lrate), m_eps(eps), m_gamma(gamma)
    {}

    void reset() override
    {
        m_history.clear();
    }

    void update(ConstAlignedMapVec& dvec, AlignedMapVec& vec) override
    {
        // Get the accumulated squared gradient associated with this gradient
        Array& grad_square = m_history[dvec.data()];

        // If length is zero, initialize it
        if (grad_square.size() == 0)
        {
            grad_square.resize(dvec.size());
            grad_square.setZero();
        }

        // Update accumulated squared gradient
        grad_square = m_gamma * grad_square + (Scalar(1) - m_gamma) *
                      dvec.array().square();
        // Update parameters
        vec.array() -= m_lrate * dvec.array() / (grad_square + m_eps).sqrt();
    }
};


} // namespace MiniDNN


#endif // MINIDNN_OPTIMIZER_RMSPROP_H_
