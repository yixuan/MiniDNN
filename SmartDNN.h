#ifndef SMARTDNN_H_
#define SMARTDNN_H_

#include <Eigen/Core>

#include "Config.h"

#include "Network.h"

#include "Layer.h"
#include "Layer/FullyConnected.h"

#include "Activation/ReLU.h"
#include "Activation/Identity.h"
#include "Activation/Sigmoid.h"
#include "Activation/Softmax.h"

#include "Output.h"
#include "Output/RegressionMSE.h"
#include "Output/BinaryClassEntropy.h"
#include "Output/MultiClassEntropy.h"

#include "Optimizer.h"
#include "Optimizer/SGD.h"
#include "Optimizer/AdaGrad.h"
#include "Optimizer/RMSProp.h"


#endif /* SMARTDNN_H_ */
