#ifndef OPTIMIZER_RMSPROP_H_
#define OPTIMIZER_RMSPROP_H_

#include <Eigen/Core>
#include "../Config.h"
#include "../Optimizer.h"
#include "../Utils/sparsepp.h"

namespace MiniDNN {


///
/// \ingroup Optimizers
///
/// The RMSProp algorithm
///
class RMSProp: public Optimizer
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
    Scalar m_decay;

    RMSProp() :
        m_lrate(Scalar(0.001)), m_eps(Scalar(1e-6)), m_decay(Scalar(0.9))
    {}

    void reset() { m_history.clear(); }

    void update(ConstAlignedMapVec& dvec, AlignedMapVec& vec)
    {
        // Get the accumulated squared gradient associated with this gradient
        Array& grad_square = m_history[dvec.data()];
        // If length is zero, initialize it
        if(grad_square.size() == 0)
        {
            grad_square.resize(dvec.size());
            grad_square.setZero();
        }
        // Update accumulated squared gradient
        grad_square = m_decay * grad_square + (Scalar(1) - m_decay) * dvec.array().square();
        // Update parameters
        vec.array() -= m_lrate * dvec.array() / (grad_square + m_eps).sqrt();
    }
};


} // namespace MiniDNN


#endif /* OPTIMIZER_RMSPROP_H_ */
