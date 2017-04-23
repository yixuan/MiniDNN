#ifndef UTILS_RANDOM_H_
#define UTILS_RANDOM_H_

#include "../Config.h"
#include <cmath>

// Shuffle the integer array
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

// Fill array with N(mu, sigma^2) random numbers
inline void set_normal_random(Scalar* arr, const int n, RNGType& rng,
		                      const Scalar& mu = Scalar(0),
							  const Scalar& sigma = Scalar(1))
{
	// For simplicity we use Box-Muller transform to generate normal random variates
	const double two_pi = 6.283185307179586476925286766559;
	for(int i = 0; i < n - 1; i += 2)
	{
		const double t1 = sigma * std::sqrt(-2 * std::log(rng.rand()));
		const double t2 = two_pi * rng.rand();
		arr[i]     = t1 * std::cos(t2) + mu;
		arr[i + 1] = t1 * std::sin(t2) + mu;
	}
	if(n % 2 == 1)
	{
		const double t1 = sigma * std::sqrt(-2 * std::log(rng.rand()));
		const double t2 = two_pi * rng.rand();
		arr[n - 1] = t1 * std::cos(t2) + mu;
	}
}


#endif /* UTILS_RANDOM_H_ */
