#ifndef __TENSOR_BF16_H__
#define __TENSOR_BF16_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

uint16_t bf16_dot_vector(uint16_t *x, uint16_t *w, int64_t d);

#ifdef __cplusplus
}
#endif

#endif