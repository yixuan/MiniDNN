#ifndef MINIDNN_OPTIMIZER_SGD_H_
#define MINIDNN_OPTIMIZER_SGD_H_

#include <Eigen/Core>
#include "../Config.h"
#include "../Optimizer.h"

namespace MiniDNN
{


///
/// \ingroup Optimizers
///
/// The Stochastic Gradient Descent (SGD) algorithm.
///
class SGD: public Optimizer
{
private:
    using Optimizer::Vector;
    using Optimizer::ConstAlignedMapVec;
    using Optimizer::AlignedMapVec;

public:
    Scalar m_lrate;
    Scalar m_decay;

    SGD(Scalar lrate = Scalar(0.001), Scalar decay = Scalar(0)) :
        m_lrate(lrate), m_decay(decay)
    {}

    void update(ConstAlignedMapVec& dvec, AlignedMapVec& vec) override
    {
        vec.noalias() -= m_lrate * (dvec + m_decay * vec);
    }
};


} // namespace MiniDNN


#endif // MINIDNN_OPTIMIZER_SGD_H_
