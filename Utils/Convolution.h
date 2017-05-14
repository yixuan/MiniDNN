#ifndef UTILS_CONVOLUTION_H_
#define UTILS_CONVOLUTION_H_

#include <Eigen/Core>
#include <vector>
#include "../Config.h"

// Convert a 1D vector into a 4D tensor
// The first two subscripts are implemented using std::vector, and the next two
// dimensions use mapped matrices. Memory layout is as follows:
// t[0, 0, , ,] => t[0, 1, , , ] => ... => t[0, d2 - 1, , , ] => t[1, 0, , , ] => t[1, 1, , , ] => ...
//
// MapMatType is either a writable mapped mat (Map<Matrix>) or a read-only one (Map<const Matrix>)
template <typename MapMatType>
void vector_to_tensor_4d(
    typename MapMatType::PointerType src, const int d1, const int d2, const int d3, const int d4,
    std::vector< std::vector<MapMatType> >& tensor
)
{
    const int inner_mat_size = d3 * d4;

    tensor.clear();
    tensor.reserve(d1);
    for(int i = 0; i < d1; i++)
    {
        tensor.push_back(std::vector<MapMatType>());
        tensor[i].reserve(d2);
        for(int j = 0; j < d2; j++)
        {
            typename MapMatType::PointerType slice_start = src + (i * d2 + j) * inner_mat_size;
            tensor[i].push_back(MapMatType(slice_start, d3, d4));
        }
    }
}

template <typename MapMatType>
void vector_to_tensor_3d(
    typename MapMatType::PointerType src, const int d1, const int d2, const int d3,
    std::vector<MapMatType>& tensor
)
{
    const int inner_mat_size = d2 * d3;

    tensor.clear();
    tensor.reserve(d1);
    for(int i = 0; i < d1; i++)
    {
        typename MapMatType::PointerType slice_start = src + i * inner_mat_size;
        tensor.push_back(MapMat(slice_start, d2, d3));
    }
}


#endif /* UTILS_CONVOLUTION_H_ */
