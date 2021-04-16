#ifndef UTILS_FACTORY_H_
#define UTILS_FACTORY_H_

#include <string>
#include <map>
#include <stdexcept>
#include "../Config.h"
#include "IO.h"
#include "Enum.h"

#include "../Layer.h"
#include "../Layer/FullyConnected.h"
#include "../Layer/Convolutional.h"
#include "../Layer/MaxPooling.h"

#include "../Activation/ReLU.h"
#include "../Activation/Sigmoid.h"
#include "../Activation/Softmax.h"
#include "../Activation/Tanh.h"
#include "../Activation/Mish.h"

#include "../Output.h"
#include "../Output/RegressionMSE.h"
#include "../Output/BinaryClassCrossEntropy.h"
#include "../Output/MultiClassCrossEntropy.h"

namespace MiniDNN
{

namespace internal
{


// Create a layer from the network meta information and the index of the layer
inline Layer* create_layer(const std::map<std::string, int>& map, int index)
{
    std::string ind = std::to_string(index);
    const int lay_id = map.find("Layer" + ind)->second;
    const LayerEnum layer_type = static_cast<LayerEnum>(lay_id);
    Layer* layer;

    if (layer_type == LayerEnum::FullyConnected)
    {
        const int in_size = map.find("in_size" + ind)->second;
        const int out_size = map.find("out_size" + ind)->second;
        layer = new FullyConnected(in_size, out_size);

    }
    else if (layer_type == LayerEnum::Convolutional)
    {
        const int in_width = map.find("in_width" + ind)->second;
        const int in_height = map.find("in_height" + ind)->second;
        const int in_channels = map.find("in_channels" + ind)->second;
        const int out_channels = map.find("out_channels" + ind)->second;
        const int window_width = map.find("window_width" + ind)->second;
        const int window_height = map.find("window_height" + ind)->second;
        layer = new Convolutional(in_width, in_height, in_channels, out_channels, window_width, window_height);
    }
    else if (layer_type == LayerEnum::MaxPooling)
    {
        const int in_width = map.find("in_width" + ind)->second;
        const int in_height = map.find("in_height" + ind)->second;
        const int in_channels = map.find("in_channels" + ind)->second;
        const int pooling_width = map.find("pooling_width" + ind)->second;
        const int pooling_height = map.find("pooling_height" + ind)->second;
        layer = new MaxPooling(in_width, in_height, in_channels, pooling_width, pooling_height);
    }
    else if (layer_type == LayerEnum::ReLU)
    {
        layer = new ReLU();
    }
    else if (layer_type == LayerEnum::Sigmoid)
    {
        layer = new Sigmoid();
    }
    else if (layer_type == LayerEnum::Softmax)
    {
        layer = new Softmax();
    }
    else if (layer_type == LayerEnum::Tanh)
    {
        layer = new Tanh();
    }
    else if (layer_type == LayerEnum::Mish)
    {
        layer = new ReLU();
    }
    else
        throw std::invalid_argument("[function create_layer]: Layer is not of a known type");

    layer->init();
    return layer;
}

// Create an output layer from the network meta information
inline Output* create_output(const std::map<std::string, int>& map)
{
    Output* output;
    int out_id = map.find("OutputLayer")->second;
    const OutputEnum out_type = static_cast<OutputEnum>(out_id);

    switch (out_type)
    {
    case OutputEnum::RegressionMSE:
        return new RegressionMSE();
    case OutputEnum::BinaryClassCrossEntropy:
        return new BinaryClassCrossEntropy();
    case OutputEnum::MultiClassCrossEntropy:
        return new MultiClassCrossEntropy();
    default:
        throw std::invalid_argument("[function create_output]: Output is not of a known type");
    }

    return output;
}


} // namespace internal

} // namespace MiniDNN


#endif /* UTILS_FACTORY_H_ */
