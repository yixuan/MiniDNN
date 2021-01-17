#ifndef OUTPUT_H_
#define OUTPUT_H_

#include <Eigen/Core>
#include <stdexcept>
#include "Config.h"

namespace MiniDNN
{


///
/// \defgroup Outputs Output Layers
///

///
/// \ingroup Outputs
///
/// The interface of the output layer of a neural network model. The output
/// layer is a special layer that associates the last hidden layer with the
/// target response variable.
///
class Output
{
    protected:
        typedef Eigen::RowVectorXi IntegerVector;

    public:
        virtual ~Output() {}

        // Check the format of target data, e.g. in classification problems the
        // target data should be binary (either 0 or 1)
        virtual void check_target_data(const Matrix& target) {}

        // Another type of target data where each element is a class label
        // This version may not be sensible for regression tasks, so by default
        // we raise an exception
        virtual void check_target_data(const IntegerVector& target)
        {
            throw std::invalid_argument("[class Output]: This output type cannot take class labels as target data");
        }

        // A combination of the forward stage and the back-propagation stage for the output layer
        // The computed derivative of the input should be stored in this layer, and can be retrieved by
        // the backprop_data() function
        virtual void evaluate(const Matrix& prev_layer_data, const Matrix& target) = 0;

        // Another type of target data where each element is a class label
        // This version may not be sensible for regression tasks, so by default
        // we raise an exception
        virtual void evaluate(const Matrix& prev_layer_data,
                              const IntegerVector& target)
        {
            throw std::invalid_argument("[class Output]: This output type cannot take class labels as target data");
        }

        // The derivative of the input of this layer, which is also the derivative
        // of the output of previous layer
        virtual const Matrix& backprop_data() const = 0;

        // Return the loss function value after the evaluation
        // This function can be assumed to be called after evaluate(), so that it can make use of the
        // intermediate result to save some computation
        virtual Scalar loss() const = 0;

        // Return the output layer type. It is used to export the NN model.
        virtual std::string output_type() const = 0;
};


} // namespace MiniDNN


#endif /* OUTPUT_H_ */
