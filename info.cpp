#include "info.h"

// code from
// https://stackoverflow.com/questions/6121792/how-to-check-if-a-cpu-supports-the-sse3-instruction-set

#ifdef _WIN32

#include <intrin.h>

//  Windows
#define cpuid(info, x) __cpuidex(info, x, 0)

#else

//  GCC Intrinsics
#include <cpuid.h>
void cpuid(int info[4], int InfoType) {
  __cpuid_count(InfoType, 0, info[0], info[1], info[2], info[3]);
}

#endif

void cpu_info(int info[4], int *n_ids, int *n_ext_ids) {
  cpuid(info, 0);
  *n_ids = info[0];
  cpuid(info, 0x80000000);
  *n_ext_ids = info[0];
}

bool has_avx() {
  int info[4];
  int n_ids;
  int n_ext_ids;
  cpu_info(info, &n_ids, &n_ext_ids);
  if (n_ids >= 0x00000001) {
    cpuid(info, 0x00000001);
    return (info[2] & ((int)1 << 28)) != 0;
  }
  return false;
}

bool has_avx2() {
  int info[4];
  int n_ids;
  int n_ext_ids;
  cpu_info(info, &n_ids, &n_ext_ids);
  if (n_ids >= 0x00000007) {
    cpuid(info, 0x00000007);
    return (info[1] & ((int)1 << 5)) != 0;
  }
  return false;
}

bool has_avx512() {
  int info[4];
  int n_ids;
  int n_ext_ids;
  cpu_info(info, &n_ids, &n_ext_ids);
  if (n_ids >= 0x00000007) {
    cpuid(info, 0x00000007);
    return (info[1] & ((int)1 << 16)) != 0;
  }
  return false;
}

bool has_sse3() {
  int info[4];
  int n_ids;
  int n_ext_ids;
  cpu_info(info, &n_ids, &n_ext_ids);
  if (n_ids >= 0x00000001) {
    cpuid(info, 0x00000001);
    return (info[2] & ((int)1 << 0)) != 0;
  }
  return false;
}

bool has_ssse3() {
  int info[4];
  int n_ids;
  int n_ext_ids;
  cpu_info(info, &n_ids, &n_ext_ids);
  if (n_ids >= 0x00000001) {
    cpuid(info, 0x00000001);
    return (info[2] & ((int)1 << 9)) != 0;
  }
  return false;
}

bool has_fma3() {
  int info[4];
  int n_ids;
  int n_ext_ids;
  cpu_info(info, &n_ids, &n_ext_ids);
  if (n_ids >= 0x00000001) {
    cpuid(info, 0x00000001);
    return (info[2] & ((int)1 << 12)) != 0;
  }
  return false;
}