#ifndef __TENSOR_F16_H__
#define __TENSOR_F16_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

bool f16_init();
uint16_t f16_dot_vector(uint16_t *x, uint16_t *w, int64_t d);

#ifdef __cplusplus
}
#endif

#endif