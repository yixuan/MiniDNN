#ifndef UTILS_RANDOM_H_
#define UTILS_RANDOM_H_

#include <Eigen/Core>
#include "../Config.h"

namespace MiniDNN
{

namespace internal
{


template <typename DerivedX, typename DerivedY, typename XType, typename YType>
inline int create_shuffled_batches(
    const Eigen::MatrixBase<DerivedX>& x, const Eigen::MatrixBase<DerivedY>& y,
    int batch_size, RNG& rng,
    std::vector<XType>& x_batches, std::vector<YType>& y_batches
)
{
    const int nobs = x.cols();
    const int dimx = x.rows();
    const int dimy = y.rows();

    if (y.cols() != nobs)
    {
        throw std::invalid_argument("Input X and Y have different number of observations");
    }

    // Randomly shuffle the IDs
    Eigen::VectorXi id = Eigen::VectorXi::LinSpaced(nobs, 0, nobs - 1);
    std::shuffle(id.data(), id.data() + id.size(), rng);

    // Compute batch size
    if (batch_size > nobs)
    {
        batch_size = nobs;
    }

    const int nbatch = (nobs - 1) / batch_size + 1;
    const int last_batch_size = nobs - (nbatch - 1) * batch_size;
    // Create shuffled data
    x_batches.clear();
    y_batches.clear();
    x_batches.reserve(nbatch);
    y_batches.reserve(nbatch);

    for (int i = 0; i < nbatch; i++)
    {
        const int bsize = (i == nbatch - 1) ? last_batch_size : batch_size;
        x_batches.push_back(XType(dimx, bsize));
        y_batches.push_back(YType(dimy, bsize));
        // Copy data
        const int offset = i * batch_size;

        for (int j = 0; j < bsize; j++)
        {
            x_batches[i].col(j).noalias() = x.col(id[offset + j]);
            y_batches[i].col(j).noalias() = y.col(id[offset + j]);
        }
    }

    return nbatch;
}


} // namespace internal

} // namespace MiniDNN


#endif /* UTILS_RANDOM_H_ */
