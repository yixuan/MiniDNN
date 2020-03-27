#ifndef OPTIMIZER_ADAM_H_
#define OPTIMIZER_ADAM_H_

#include <Eigen/Core>
#include <map>
#include "../Config.h"
#include "../Optimizer.h"

namespace MiniDNN
{


///
/// \ingroup Optimizers
///
/// The Adam algorithm
///
class Adam: public Optimizer
{
    private:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef Eigen::Array<Scalar, Eigen::Dynamic, 1> Array;
        typedef Vector::ConstAlignedMapType ConstAlignedMapVec;
        typedef Vector::AlignedMapType AlignedMapVec;

        std::map<const Scalar*, Array> m_history_m;
        std::map<const Scalar*, Array> m_history_v;
        Scalar m_beta1t;
        Scalar m_beta2t;

    public:
        Scalar m_lrate;
        Scalar m_eps;
        Scalar m_beta1;
        Scalar m_beta2;

        Adam(const Scalar& lrate = Scalar(0.001), const Scalar& eps = Scalar(1e-6),
             const Scalar& beta1 = Scalar(0.9), const Scalar& beta2 = Scalar(0.999)) :
            m_beta1t(beta1), m_beta2t(beta2),
            m_lrate(lrate), m_eps(eps),
            m_beta1(beta1), m_beta2(beta2)
        {}

        void reset()
        {
            m_history_m.clear();
            m_history_v.clear();
            m_beta1t = m_beta1;
            m_beta2t = m_beta2;
        }

        // https://ruder.io/optimizing-gradient-descent/index.html
        void update(ConstAlignedMapVec& dvec, AlignedMapVec& vec)
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


#endif /* OPTIMIZER_ADAM_H_ */
