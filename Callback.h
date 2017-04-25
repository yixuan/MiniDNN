#ifndef CALLBACK_H_
#define CALLBACK_H_

#include <Eigen/Core>
#include "Config.h"

class Network;

class Callback
{
protected:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::RowVectorXi IntegerVector;

public:
    // Public members that will be set by the network during the training process
    int m_nbatch;   // Number of total batches
    int m_batch_id; // The index for the current mini-batch (0, 1, ..., m_nbatch-1)
    int m_nepoch;   // Total number of epochs (one run on the whole data set) in the training process
    int m_epoch_id; // The index for the current epoch (0, 1, ..., m_nepoch-1)

    Callback() :
        m_nbatch(0), m_batch_id(0), m_nepoch(0), m_epoch_id(0)
    {}

    virtual ~Callback() {}

    // Before training a mini-batch
    virtual void pre_training_batch(const Network* net, const Matrix& x, const Matrix& y) {}
    virtual void pre_training_batch(const Network* net, const Matrix& x, const IntegerVector& y) {}

    // After a mini-batch is trained
    virtual void post_training_batch(const Network* net, const Matrix& x, const Matrix& y) {}
    virtual void post_training_batch(const Network* net, const Matrix& x, const IntegerVector& y) {}
};


#endif /* CALLBACK_H_ */
