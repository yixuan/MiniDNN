#ifndef UTILS_IO_H_
#define UTILS_IO_H_

#include <string>   // std::string
#include <sstream>  // std::ostringstream

#ifdef _WIN32
	#include <windows.h>  // _mkdir
#else
	#include <sys/stat.h> // mkdir
#endif

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
inline std::string to_string(const NumberType& num)
{
    std::ostringstream convert;
    convert << num;
    return convert.str();
}

///
/// Create a directory
///
/// \param dir     Name of the directory to be created
/// \return        \c true if the directory is successfully created
///
inline bool create_directory(const std::string& dir)
{
#ifdef _WIN32
	return 0 == _mkdir(dir.c_str());
#else
	return 0 == mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
#endif
}


} // namespace internal

} // namespace MiniDNN


#endif /* UTILS_IO_H_ */
