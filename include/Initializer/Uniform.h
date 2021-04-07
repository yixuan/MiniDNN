#ifndef MINIDNN_INITIALIZER_UNIFORM_H_
#define MINIDNN_INITIALIZER_UNIFORM_H_

#include <random>
#include <Eigen/Core>
#include "../Config.h"
#include "../Initializer.h"

namespace MiniDNN
{


///
/// \ingroup Initializer
///
/// Initialize parameters using uniform distribution
///
class Uniform: Initializer
{
private:
    using Index = Eigen::Index;
    using Initializer::GenericMatrix;

    const Scalar m_a;
    const Scalar m_b;

public:
    Uniform(Scalar a = Scalar(0), Scalar b = Scalar(1)) :
        m_a(std::min(a, b)), m_b(std::max(a, b))
    {}

    void initialize(GenericMatrix mat, RNG& rng)
    {
        std::uniform_real_distribution<Scalar> unif(m_a, m_b);
        Scalar* start = mat.data();
        Index len = mat.size();
        for (Index i = 0; i < len; i++)
            start[i] = unif(rng);
    };
};


} // namespace MiniDNN


#endif // MINIDNN_INITIALIZER_UNIFORM_H_
