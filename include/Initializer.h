#ifndef MINIDNN_INITIALIZER_H_
#define MINIDNN_INITIALIZER_H_

#include <Eigen/Core>
#include "Config.h"

namespace MiniDNN
{


///
/// \defgroup Initializer Initialization Methods
///

///
/// \ingroup Initializer
///
/// The interface of initialization methods.
///
class Initializer
{
protected:
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using GenericMatrix = Eigen::Ref<Matrix>;

public:
    ///
    /// Virtual destructor.
    ///
    virtual ~Initializer() {}

    ///
    /// Initialize the given matrix or vector.
    ///
    virtual void initialize(GenericMatrix mat, RNG& rng) const = 0;
};


} // namespace MiniDNN


#endif // MINIDNN_INITIALIZER_H_
