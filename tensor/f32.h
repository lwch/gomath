#ifndef __TENSOR_F32_H__
#define __TENSOR_F32_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

float f32_dot_vector(float *x, float *w, int64_t d);

#ifdef __cplusplus
}
#endif

#endif