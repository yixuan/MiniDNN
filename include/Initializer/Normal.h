#ifndef MINIDNN_INITIALIZER_NORMAL_H_
#define MINIDNN_INITIALIZER_NORMAL_H_

#include <random>
#include <Eigen/Core>
#include "../Config.h"
#include "../Initializer.h"

namespace MiniDNN
{


///
/// \ingroup Initializer
///
/// Initialize parameters using normal distribution
///
class Normal: Initializer
{
private:
    using Index = Eigen::Index;
    using Initializer::GenericMatrix;

    const Scalar m_mu;
    const Scalar m_sigma;

public:
    ///
    /// Constructor for a N(mu, sigma^2) distribution initializer
    ///
    Normal(Scalar mu = Scalar(0), Scalar sigma = Scalar(0.1)) :
        m_mu(mu), m_sigma(sigma)
    {}

    ///
    /// Initialize the given matrix or vector with a N(mu, sigma^2) distribution
    ///
    void initialize(GenericMatrix mat, RNG& rng)
    {
        std::normal_distribution<Scalar> normal(m_mu, m_sigma);
        Scalar* start = mat.data();
        Index len = mat.size();
        for (Index i = 0; i < len; i++)
            start[i] = normal(rng);
    };
};


} // namespace MiniDNN


#endif // MINIDNN_INITIALIZER_NORMAL_H_
