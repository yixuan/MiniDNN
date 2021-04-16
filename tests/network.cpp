#include <MiniDNN.h>
#include "catch.hpp"

using namespace MiniDNN;

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

// Learning the XOR function using MSE loss
TEST_CASE("Feed-forward network with MSE loss", "[fnn_mse]")
{
    // Set random seed
    RNG rng(123);
    // Predictors -- each column is an observation
    Matrix x(2, 4);
    x << 0, 0, 1, 1,
         0, 1, 0, 1;
    // Response variables -- each column is an observation
    // y = XOR(x1, x2)
    Matrix y(1, 4);
    y << 0, 1, 1, 0;

    // Construct a network object
    Network net(rng);

    // Create three layers
    net.add_layer(FullyConnected(2, 10));
    net.add_layer(ReLU());
    net.add_layer(FullyConnected(10, 1));
    net.add_layer(Sigmoid());

    // Set output layer
    net.set_output(RegressionMSE());

    // Create optimizer object
    RMSProp opt;
    opt.m_lrate = 0.1;

    // (Optional) set callback function object
    net.set_callback(VerboseCallback());

    // Initialize parameters with N(0, 0.01^2) using random seed 123
    Normal initializer(0.0, 0.01);
    net.init(initializer);

    // Fit the model with a batch size of 4, running 100 epochs
    net.fit(opt, x, y, 4, 100);

    // Obtain prediction -- each column is an observation
    Matrix pred = net.predict(x);
    std::cout << "prediction = " << pred << std::endl;

    REQUIRE((pred - y).cwiseAbs().maxCoeff() < 0.1);
}

TEST_CASE("Feed-forward network with binary cross entropy", "[fnn_bce]")
{
    // Set random seed
    RNG rng(123);
    // Predictors -- each column is an observation
    Matrix x(2, 4);
    x << 0, 0, 1, 1,
         0, 1, 0, 1;
    // Response variables -- each column is an observation
    // y = XOR(x1, x2)
    Matrix y(1, 4);
    y << 0, 1, 1, 0;

    // Construct a network object
    Network net(rng);

    // Create three layers
    net.add_layer(FullyConnected(2, 10));
    net.add_layer(ReLU());
    net.add_layer(FullyConnected(10, 1));
    net.add_layer(Sigmoid());

    // Set output layer
    net.set_output(BinaryClassEntropy());

    // Create optimizer object
    RMSProp opt;
    opt.m_lrate = 0.1;

    // (Optional) set callback function object
    net.set_callback(VerboseCallback());

    // Initialize parameters with N(0, 0.01^2) using random seed 123
    Normal initializer(0.0, 0.01);
    net.init(initializer);

    // Fit the model with a batch size of 4, running 100 epochs
    net.fit(opt, x, y, 4, 100);

    // Obtain prediction -- each column is an observation
    Matrix pred = net.predict(x);
    std::cout << "prediction = " << pred << std::endl;

    REQUIRE((pred - y).cwiseAbs().maxCoeff() < 0.1);
}
