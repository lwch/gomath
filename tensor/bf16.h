#ifndef __TENSOR_BF16_H__
#define __TENSOR_BF16_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

void bf16_init();
float bf16_dot_vector(float *x, float *w, int64_t d);

#ifdef __cplusplus
}
#endif

#endif