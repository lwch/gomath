#ifndef __TENSOR_FP16_H__
#define __TENSOR_FP16_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

bool fp16_init();
uint16_t fp16_dot_vector(uint16_t *x, uint16_t *w, int64_t d);
void fp16_mul_vector(uint16_t *x, uint16_t *w, uint16_t *y, int64_t d);

#ifdef __cplusplus
}
#endif

#endif