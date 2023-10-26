#ifndef __TENSOR_TENSOR_H__
#define __TENSOR_TENSOR_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>

extern size_t bs; // batch size of simd

bool tensor_init();

#ifdef __cplusplus
}
#endif

#endif