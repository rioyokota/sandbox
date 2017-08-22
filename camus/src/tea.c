#include <stdint.h>
void encrypt_(uint32_t* v) {
uint32_t v0=v[0], v1=v[1], sum=0, i; /* set up */
uint32_t delta=0x9e3779b9; /* a key schedule constant */

uint32_t k0=0xA341316C, k1=0xC8013EA4, k2=0xAD90777D, k3=0x7E95761E;
// key value : 10.1016/j.jcp.2011.05.021 " For the TEA8 hash-based PRNG, we use the key provided by reference [26],or{k1 = 0xA341316C, k2 = 0xC8013EA4, k3 = 0xAD90777D, k4 = 0x7E95761E}. "

//for (i=0; i < 32; i++) { /* basic cycle start */
//TEA with eight rounds (TEA8) as recommended by reference [26]. >> http://dx.doi.org/10.1016/j.jcp.2011.05.021
//[26] F. Zafar, A. Curtis, M. Olano, GPU random numbers via the tiny encryption algorithm, in: HPG 2010: Proceedings of the ACM SIGGRAPH/Eurographics Symposium on High Performance Graphics, June 2009.
for (i=0; i < 8; i++) { /* basic cycle start */
sum += delta;
v0 += ((v1<<4) + k0) ^ (v1 + sum) ^ ((v1>>5) + k1);
v1 += ((v0<<4) + k2) ^ (v0 + sum) ^ ((v0>>5) + k3);
} /* end cycle */
v[0]=v0; v[1]=v1;
}
