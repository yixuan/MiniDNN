#ifndef UTILS_DEFAULTRNG_H_
#define UTILS_DEFAULTRNG_H_

// Based on public domain code by Ray Gardner
// http://stjarnhimlen.se/snippets/rg_rand.c
class DefaultRNG
{
private:
    const int m_a;     // multiplier
    const long m_max;  // 2^31 - 1
    long m_rand;

    inline long next_long_rand(long seed)
    {
        unsigned long lo, hi;

        lo = m_a * (long)(seed & 0xFFFF);
        hi = m_a * (long)((unsigned long)seed >> 16);
        lo += (hi & 0x7FFF) << 16;
        if((long)lo > m_max)
        {
            lo &= m_max;
            ++lo;
        }
        lo += hi >> 15;
        if((long)lo > m_max)
        {
            lo &= m_max;
            ++lo;
        }
        return (long)lo;
    }
public:
    DefaultRNG(unsigned long init_seed) :
        m_a(16807),
        m_max(2147483647L),
        m_rand(init_seed ? (init_seed & m_max) : 1)
    {}

    void seed(unsigned long seed)
    {
    	m_rand = (seed ? (seed & m_max) : 1);
    }

    double rand()
    {
        m_rand = next_long_rand(m_rand);
        return double(m_rand) / double(m_max);
    }
};


#endif /* UTILS_DEFAULTRNG_H_ */
