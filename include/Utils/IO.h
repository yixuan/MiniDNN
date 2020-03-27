#ifndef UTILS_IO_H_
#define UTILS_IO_H_

#include <string>   // std::string
#include <sstream>  // std::ostringstream
#include "../Config.h"

namespace MiniDNN
{

namespace internal
{


///
/// Convert a number to a string in C++98
///
/// \tparam NumberType     Type of the number
/// \param  num            The number to be converted
/// \return                An std::string containing the number
///
template <class NumberType>
std::string to_string(const NumberType& num)
{
    std::ostringstream convert;
    convert << num;
    return convert.str();
}


} // namespace internal

} // namespace MiniDNN


#endif /* UTILS_IO_H_ */
