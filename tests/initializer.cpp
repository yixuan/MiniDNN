#include <Eigen/Core>
#include <Initializer/Uniform.h>
#include <Initializer/Normal.h>
#include "catch.hpp"

using namespace MiniDNN;
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

TEST_CASE("Uniform initializer", "[uniform]")
{
    Matrix mat(3, 5);
    Vector vec(10);

    // Uniform initializer
    Uniform init(-0.5, 0.5);
    // RNG with seed 123
    RNG rng(123);

    init.initialize(mat, rng);
    INFO("\nInitialized matrix:\n" << mat);

    init.initialize(vec, rng);
    INFO("\nInitialized vector:\n" << vec.transpose());

    // We just want to make sure that this code compiles
    REQUIRE(true);
}

TEST_CASE("Normal initializer", "[normal]")
{
    Matrix mat(3, 5);
    Vector vec(10);

    // Normal initializer
    Normal init(0.0, 1.0);
    // RNG with seed 123
    RNG rng(123);

    init.initialize(mat, rng);
    INFO("\nInitialized matrix:\n" << mat);

    init.initialize(vec, rng);
    INFO("\nInitialized vector:\n" << vec.transpose());

    // We just want to make sure that this code compiles
    REQUIRE(true);
}
