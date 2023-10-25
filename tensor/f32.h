#ifndef __TENSOR_F32_H__
#define __TENSOR_F32_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

bool f32_init();
float f32_dot_vector(float *x, float *w, int64_t d);

#ifdef __cplusplus
}
#endif

#endif