#ifndef MINIDNN_UTILS_ENUM_H_
#define MINIDNN_UTILS_ENUM_H_

#include <string>
#include <stdexcept>
#include "../Config.h"

namespace MiniDNN
{

namespace internal
{


// Enumerations for hidden layers
enum class LayerEnum
{
    FullyConnected = 0,
    Convolutional,
    MaxPooling,

    ReLU = 100,
    Sigmoid,
    Softmax,
    Tanh,
    Mish
};

// Convert a hidden layer type string to an integer
inline LayerEnum layer_id(const std::string& type)
{
    if (type == "FullyConnected")
        return LayerEnum::FullyConnected;
    if (type == "Convolutional")
        return LayerEnum::Convolutional;
    if (type == "MaxPooling")
        return LayerEnum::MaxPooling;

    if (type == "ReLU")
        return LayerEnum::ReLU;
    if (type == "Sigmoid")
        return LayerEnum::Sigmoid;
    if (type == "Softmax")
        return LayerEnum::Softmax;
    if (type == "Tanh")
        return LayerEnum::Tanh;
    if (type == "Mish")
        return LayerEnum::Mish;

    throw std::invalid_argument("[function layer_id]: Layer is not of a known type");
    return LayerEnum::FullyConnected;
}

// Enumerations for output layers
enum class OutputEnum
{
    RegressionMSE = 0,
    BinaryClassCrossEntropy,
    MultiClassCrossEntropy
};

// Convert an output layer type string to an integer
inline OutputEnum output_id(const std::string& type)
{
    if (type == "RegressionMSE")
        return OutputEnum::RegressionMSE;
    if (type == "BinaryClassCrossEntropy")
        return OutputEnum::BinaryClassCrossEntropy;
    if (type == "MultiClassCrossEntropy")
        return OutputEnum::MultiClassCrossEntropy;

    throw std::invalid_argument("[function output_id]: Output is not of a known type");
    return OutputEnum::RegressionMSE;
}


} // namespace internal

} // namespace MiniDNN


#endif // MINIDNN_UTILS_ENUM_H_
