#ifndef MINIDNN_OPTIMIZER_ADAM_H_
#define MINIDNN_OPTIMIZER_ADAM_H_

#include <Eigen/Core>
#include <unordered_map>
#include "../Config.h"
#include "../Optimizer.h"

namespace MiniDNN
{


///
/// \ingroup Optimizers
///
/// The Adam algorithm.
///
class Adam: public Optimizer
{
private:
    using Array = Eigen::Array<Scalar, Eigen::Dynamic, 1>;
    using Optimizer::ConstAlignedMapVec;
    using Optimizer::AlignedMapVec;

    std::unordered_map<const Scalar*, Array> m_history_m;
    std::unordered_map<const Scalar*, Array> m_history_v;
    Scalar m_beta1t;
    Scalar m_beta2t;

public:
    Scalar m_lrate;
    Scalar m_eps;
    Scalar m_beta1;
    Scalar m_beta2;

    Adam(Scalar lrate = Scalar(0.001), Scalar eps = Scalar(1e-6),
         Scalar beta1 = Scalar(0.9), Scalar beta2 = Scalar(0.999)) :
        m_beta1t(beta1), m_beta2t(beta2),
        m_lrate(lrate), m_eps(eps),
        m_beta1(beta1), m_beta2(beta2)
    {}

    void reset() override
    {
        m_history_m.clear();
        m_history_v.clear();
        m_beta1t = m_beta1;
        m_beta2t = m_beta2;
    }

    // https://ruder.io/optimizing-gradient-descent/index.html
    void update(ConstAlignedMapVec& dvec, AlignedMapVec& vec) override
    {
        using std::sqrt;
        // Get the m and v vectors associated with this gradient
        Array& mvec = m_history_m[dvec.data()];
        Array& vvec = m_history_v[dvec.data()];

        // If length is zero, initialize it
        if (mvec.size() == 0)
        {
            mvec.resize(dvec.size());
            mvec.setZero();
        }

        if (vvec.size() == 0)
        {
            vvec.resize(dvec.size());
            vvec.setZero();
        }

        // Update m and v vectors
        mvec = m_beta1 * mvec + (Scalar(1) - m_beta1) * dvec.array();
        vvec = m_beta2 * vvec + (Scalar(1) - m_beta2) * dvec.array().square();
        // Correction coefficients
        const Scalar correct1 = Scalar(1) / (Scalar(1) - m_beta1t);
        const Scalar correct2 = Scalar(1) / sqrt(Scalar(1) - m_beta2t);
        // Update parameters
        vec.array() -= (m_lrate * correct1) * mvec / (correct2 * vvec.sqrt() + m_eps);
        m_beta1t *= m_beta1;
        m_beta2t *= m_beta2;
    }
};


} // namespace MiniDNN


#endif // MINIDNN_OPTIMIZER_ADAM_H_
