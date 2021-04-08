#include <Eigen/Core>
#include <Initializer/Normal.h>  // To generate random numbers
#include <Activation/Mish.h>
#include <Activation/ReLU.h>
#include <Activation/Sigmoid.h>
#include <Activation/Tanh.h>
#include "catch.hpp"

using namespace MiniDNN;
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

// ================== The scalar version of the activation function ==================
template <typename Act>
inline double act_fun(double x);

template <>
inline double act_fun<ReLU>(double x)
{
    return std::max(x, 0.0);
}

template <>
inline double act_fun<Sigmoid>(double x)
{
    return 1.0 / (1.0 + std::exp(-x));
}

template <>
inline double act_fun<Tanh>(double x)
{
    return std::tanh(x);
}

template <>
inline double act_fun<Mish>(double x)
{
    return x * std::tanh(std::log(1.0 + std::exp(x)));
}
// ===================================================================================

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

template <typename Act>
void check_activation(const Matrix& z, Scalar tol = Scalar(1e-12))
{
    const int d = z.rows();
    const int n = z.cols();
    INFO("\nInput matrix:\n" << z);

    // Forward pass, a = f(z)
    Act act;
    act.forward(z);
    const Matrix& a = act.output();
    INFO("\nActivated matrix:\n" << a);

    // Compute the forward result using the scalar version
    Matrix a_true(d, n);
    for(int j = 0; j < n; j++)
        for(int i = 0; i < d; i++)
            a_true(i, j) = act_fun<Act>(z(i, j));

    REQUIRE((a - a_true).cwiseAbs().maxCoeff() == Approx(0.0).margin(tol));

    // l = sum(a[i, j]^2)
    // dl/da = 2*a
    Matrix dlda = 2.0 * a;
    act.backprop(z, dlda);
    const Matrix& dldz = act.backprop_data();
    INFO("\nGradient of input:\n" << dldz);

    // Compute the gradient using numerical differentiation
    Matrix dldz_approx(d, n);
    const double eps = 1e-6;
    for(int j = 0; j < n; j++)
    {
        for(int i = 0; i < d; i++)
        {
            const double z1 = z(i, j) - eps;
            const double z2 = z1 + 2.0 * eps;
            const double f1 = act_fun<Act>(z1);
            const double f2 = act_fun<Act>(z2);
            const double df = 0.5 * (f2 - f1) / eps;
            dldz_approx(i, j) = 2.0 * a_true(i, j) * df;
        }
    }

    REQUIRE((dldz - dldz_approx).cwiseAbs().maxCoeff() == Approx(0.0).margin(std::sqrt(tol)));
}

TEST_CASE("Mish activation function", "[mish]")
{
    Matrix z = test_matrix();
    check_activation<Mish>(z);
}

TEST_CASE("ReLU activation function", "[relu]")
{
    Matrix z = test_matrix();
    check_activation<ReLU>(z);
}

TEST_CASE("Sigmoid activation function", "[sigmoid]")
{
    Matrix z = test_matrix();
    check_activation<Sigmoid>(z);
}

TEST_CASE("Tanh activation function", "[tanh]")
{
    Matrix z = test_matrix();
    check_activation<Tanh>(z);
}
