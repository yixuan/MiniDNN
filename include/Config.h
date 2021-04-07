#ifndef MINIDNN_CONFIG_H_
#define MINIDNN_CONFIG_H_

#include <random>

namespace MiniDNN
{


// Floating-point number type
#ifndef MINIDNN_SCALAR
using Scalar = double;
#else
using Scalar = MINIDNN_SCALAR;
#endif

// C++11 random number generator
using RNG = std::mt19937;


} // namespace MiniDNN

#endif // MINIDNN_CONFIG_H_
