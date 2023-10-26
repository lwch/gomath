#ifndef __GOMATH_INFO_H__
#define __GOMATH_INFO_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

bool has_avx();
bool has_avx512();

#ifdef __cplusplus
}
#endif

#endif