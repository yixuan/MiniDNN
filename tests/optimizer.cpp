#include <Eigen/Core>
#include <Initializer/Normal.h>  // To generate random numbers
#include <Optimizer/AdaGrad.h>
#include <Optimizer/Adam.h>
#include <Optimizer/RMSProp.h>
#include <Optimizer/SGD.h>
#include "catch.hpp"

using namespace MiniDNN;
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

TEST_CASE("Optimizers", "[optimizer]")
{
    AdaGrad adagrad;
    Adam adam;
    RMSProp rmsprop;
    SGD sgd;

    Vector vec(100), dvec(100);
    Vector::AlignedMapType mvec(vec.data(), vec.size());
    Vector::ConstAlignedMapType mdvec(dvec.data(), dvec.size());

    // Initialize vectors
    Normal init(0.0, 1.0);
    RNG rng(123);
    init.initialize(vec, rng);
    init.initialize(dvec, rng);

    // Just to make sure that the code compiles
    adagrad.update(mdvec, mvec);
    adagrad.update(mdvec, mvec);
    adagrad.update(mdvec, mvec);

    adam.update(mdvec, mvec);
    adam.update(mdvec, mvec);
    adam.update(mdvec, mvec);

    rmsprop.update(mdvec, mvec);
    rmsprop.update(mdvec, mvec);
    rmsprop.update(mdvec, mvec);

    sgd.update(mdvec, mvec);
    sgd.update(mdvec, mvec);
    sgd.update(mdvec, mvec);
    REQUIRE(true);
}
