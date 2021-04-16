#ifndef MINIDNN_H_
#define MINIDNN_H_

#include <Eigen/Core>

#include "Config.h"

#include "Initializer.h"
#include "Initializer/Normal.h"
#include "Initializer/Uniform.h"

#include "Layer.h"
#include "Layer/FullyConnected.h"
#include "Layer/Convolutional.h"
#include "Layer/MaxPooling.h"

#include "Activation/ReLU.h"
#include "Activation/Sigmoid.h"
#include "Activation/Softmax.h"
#include "Activation/Tanh.h"
#include "Activation/Mish.h"

#include "Output.h"
#include "Output/RegressionMSE.h"
#include "Output/BinaryClassCrossEntropy.h"
#include "Output/MultiClassEntropy.h"

#include "Optimizer.h"
#include "Optimizer/SGD.h"
#include "Optimizer/AdaGrad.h"
#include "Optimizer/RMSProp.h"
#include "Optimizer/Adam.h"

#include "Callback.h"
#include "Callback/VerboseCallback.h"

#include "Network.h"


#endif // MINIDNN_H_
