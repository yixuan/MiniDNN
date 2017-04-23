#ifndef UTILS_RANDOM_H_
#define UTILS_RANDOM_H_

#include "../Config.h"

inline void shuffle(int* arr, const int n, RNGType& rng)
{
	for(int i = n - 1; i > 0; i--)
	{
		// A random non-negative integer <= i
		const int j = int(rng.rand() * (i + 1));
		// Swap arr[i] and arr[j]
		const int tmp = arr[i];
		arr[i] = arr[j];
		arr[j] = tmp;
	}
}


#endif /* UTILS_RANDOM_H_ */
