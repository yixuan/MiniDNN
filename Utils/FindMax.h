#ifndef UTILS_FINDMAX_H_
#define UTILS_FINDMAX_H_

#include "../Config.h"

// Find the location of the maximum element in x[0], x[1], ..., x[n-1]
// Special cases for small n using recursive template
// N is assumed to be >= 2
template <int N>
inline int find_max(const Scalar* x)
{
    int loc = find_max<N - 1>(x);
    loc = (x[loc] < x[N - 1]) ? (N - 1) : loc;
    return loc;
}

template <>
inline int find_max<2>(const Scalar* x)
{
    return (x[0] < x[1]) ? 1: 0;
}

// n is assumed be >= 2
inline int find_max(const Scalar* x, const int n)
{
    switch(n)
    {
        case 2:
            return find_max<2>(x);
        case 3:
            return find_max<3>(x);
        case 4:
            return find_max<4>(x);
        case 5:
            return find_max<5>(x);
    }
    int loc = find_max<5>(x);
    for(int i = 5; i < n; i++)
    {
        loc = (x[loc] < x[i]) ? i : loc;
    }
    return loc;
}

// Find the maximum element in the block src[0:(nrow-1), 0:(ncol-1)]
// col_stride is the distance between src[0, 0] and src[0, 1]
// Special cases for small n
inline int find_block_max(const Scalar* src, const int nrow, const int ncol, const int col_stride)
{
    // Max elements in the first and second columns
    int loc = find_max(src, nrow);
    const Scalar* col_start = src + col_stride;
    int loc_offset = col_stride;
    int loc_next = loc_offset + find_max(col_start, nrow);
    loc = (src[loc_next] > src[loc]) ? loc_next: loc;
    if(ncol == 2)  return loc;

    // 3rd column
    col_start += col_stride;
    loc_offset += col_stride;
    loc_next = loc_offset + find_max(col_start, nrow);
    loc = (src[loc_next] > src[loc]) ? loc_next: loc;
    if(ncol == 3)  return loc;

    // 4th column
    col_start += col_stride;
    loc_offset += col_stride;
    loc_next = loc_offset + find_max(col_start, nrow);
    loc = (src[loc_next] > src[loc]) ? loc_next: loc;
    if(ncol == 4)  return loc;

    // 5th column
    col_start += col_stride;
    loc_offset += col_stride;
    loc_next = loc_offset + find_max(col_start, nrow);
    loc = (src[loc_next] > src[loc]) ? loc_next: loc;
    if(ncol == 5)  return loc;

    // Other columns
    for(int i = 5; i < ncol; i++)
    {
        col_start += col_stride;
        loc_offset += col_stride;
        loc_next = loc_offset + find_max(col_start, nrow);
        loc = (src[loc_next] > src[loc]) ? loc_next: loc;
    }
    return loc;
}

#endif /* UTILS_FINDMAX_H_ */
