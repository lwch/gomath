#ifndef __GOMATH_INFO_H__
#define __GOMATH_INFO_H__

#ifdef __cplusplus
extern "C" {
#endif

bool has_avx();
bool has_avx2();
bool has_avx512();
bool has_sse3();
bool has_ssse3();
bool has_neon();

#ifdef __cplusplus
}
#endif

#endif