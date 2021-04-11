#ifndef MINIDNN_UTILS_SHUFFLE_H_
#define MINIDNN_UTILS_SHUFFLE_H_

#include <Eigen/Core>
#include "../Config.h"

namespace MiniDNN
{

namespace internal
{


inline int create_shuffled_indices(int nobs, int batch_size, std::vector<Eigen::VectorXi>& ids, RNG& rng)
{
    // Randomly shuffle the IDs
    Eigen::VectorXi id = Eigen::VectorXi::LinSpaced(nobs, 0, nobs - 1);
    std::shuffle(id.data(), id.data() + id.size(), rng);

    // Compute nunmber of batches and batch sizes
    batch_size = std::min(nobs, batch_size);
    const int nbatch = (nobs - 1) / batch_size + 1;
    const int last_batch_size = nobs - (nbatch - 1) * batch_size;

    ids.resize(nbatch);
    for (int i = 0; i < nbatch; i++)
        ids[i] = id.segment(i * batch_size, (i == nbatch - 1) ? last_batch_size : batch_size);

    return nbatch;
}

template <typename DataType, typename Derived>
DataType subset_data(const Eigen::MatrixBase<Derived>& data, const Eigen::VectorXi& id)
{
    const int batch_size = id.size();
    const int dim = data.rows();
    DataType subdata(dim, batch_size);
    for (int i = 0; i < batch_size; i++)
        subdata.col(i).noalias() = data.col(id[i]);

    return subdata;
}


} // namespace internal

} // namespace MiniDNN


#endif // MINIDNN_UTILS_SHUFFLE_H_
