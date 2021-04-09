#include <Eigen/Core>
#include <Initializer/Normal.h>  // To generate random numbers
#include <Layer/FullyConnected.h>
#include "catch.hpp"

using namespace MiniDNN;
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

Matrix test_matrix()
{
    // Input matrix
    const int d = 3;
    const int n = 5;
    Matrix z(d, n);

    // Initialize z with a normal distribution
    Normal init(0.0, 1.0);
    RNG rng(123);
    init.initialize(z, rng);
    return z;
}

// Softmax is special
TEST_CASE("Softmax activation function", "[softmax]")
{
    const Scalar tol = 1e-12;
    Matrix x = test_matrix();
    const int d = x.rows();
    const int n = x.cols();
    INFO("\nInput matrix:\n" << x);

    // Fully-connected layer
    const int p = 2 * d;
    FullyConnected layer(d, p);

    // Initialize parameters
    Normal init(0.0, 1.0);
    RNG rng(123);
    layer.init(init, rng);

    // Save the parameters
    std::vector<double> param = layer.get_parameters();
    std::size_t nparam = param.size();

    // Forward pass, z = f(x)
    layer.forward(x);
    const Matrix& z = layer.output();
    INFO("\nOutput matrix:\n" << z);

    // l = sum(z[i, j]^2)
    // dl/dz = 2*z
    Matrix dldz = 2.0 * z;
    layer.backprop(x, dldz);

    // Compute the gradient of input and parameters
    const Matrix dldx = layer.backprop_data();
    INFO("\nGradient of input:\n" << dldx);
    std::vector<double> dldparam = layer.get_derivatives();

    // Compute the gradient of input using numerical differentiation
    Matrix dldx_approx(d, n);
    const double eps = 1e-6;
    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < d; i++)
        {
            Matrix x1 = x, x2 = x;
            x1(i, j) -= eps;
            x2(i, j) += eps;
            layer.forward(x1);
            const Matrix& z1 = layer.output();
            const double f1 = z1.cwiseAbs2().sum();
            layer.forward(x2);
            const Matrix& z2 = layer.output();
            const double f2 = z2.cwiseAbs2().sum();
            dldx_approx(i, j) = 0.5 * (f2 - f1) / eps;
        }
    }

    REQUIRE((dldx - dldx_approx).cwiseAbs().maxCoeff() == Approx(0.0).margin(10 * eps));

    // Compute the gradient of parameters using numerical differentiation
    Vector dldparam_approx(nparam);
    std::vector<double> param_eps(param);
    for (std::size_t i = 0; i < nparam; i++)
    {
        param_eps[i] = param[i] - eps;
        layer.set_parameters(param_eps);
        layer.forward(x);
        const Matrix& z1 = layer.output();
        const double f1 = z1.cwiseAbs2().sum();

        param_eps[i] = param[i] + eps;
        layer.set_parameters(param_eps);
        layer.forward(x);
        const Matrix& z2 = layer.output();
        const double f2 = z2.cwiseAbs2().sum();

        dldparam_approx[i] = 0.5 * (f2 - f1) / eps;
    }

    Eigen::Map<Vector> dldparam_v(&dldparam[0], nparam);
    REQUIRE((dldparam_v - dldparam_approx).cwiseAbs().maxCoeff() == Approx(0.0).margin(10 * eps));
}
