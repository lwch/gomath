package gomath

// #include <stdbool.h>
// #include "info.h"
import "C"

func HasAVX() bool {
	return bool(C.has_avx())
}

func HasAVX2() bool {
	return bool(C.has_avx2())
}

func HasAVX512() bool {
	return bool(C.has_avx512())
}

func HasSSE3() bool {
	return bool(C.has_sse3())
}

func HasSSSE3() bool {
	return bool(C.has_ssse3())
}

func HasFMA3() bool {
	return bool(C.has_fma3())
}
